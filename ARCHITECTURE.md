# Jarvis — Architecture cible

**Objectif** : assistant vocal entièrement local (STT → LLM → TTS), avec
tool calling, accès CRUD au système de fichiers, et connexion MCP à des
services externes.

Ce document décrit l'architecture visée et les décisions structurantes.
L'état actuel est dans [`AUDIT.md`](AUDIT.md), le chemin pour y arriver dans
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## 1. Vue d'ensemble

```
  micro ──► capture ──► VAD ──► STT ──┐
                                       │
                          ┌────────────▼────────────┐
                          │      Boucle agent       │◄── conversation
                          │  (tool calling borné)   │    (persistée)
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │    Registre d'outils    │
                          │  namespacing + sélection│
                          └────────────┬────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
       ┌────────▼────────┐   ┌─────────▼─────────┐   ┌────────▼────────┐
       │  Outils locaux  │   │  Client MCP       │   │  Moteur de      │
       │  fs CRUD · web  │   │  N serveurs       │   │  politique      │
       │  applications   │   │  stdio + HTTP     │   │  (confirmation) │
       └─────────────────┘   └───────────────────┘   └────────┬────────┘
                                                              │
  haut-parleur ◄── playback ◄── TTS ◄── découpage phrases      ▼
       ▲                                                  surfaces de
       └──────────── barge-in (interruption) ──────────    confirmation
                                                          voix · TUI · clavier
```

Tout tourne sur la machine de l'utilisateur. Les seules sorties réseau sont
celles que l'utilisateur autorise explicitement : les serveurs MCP distants
qu'il configure, et les URL qu'il approuve.

---

## 2. Découpage des modules

> **Où on en est.** Le paquet `src/jarvis/` et le point d'entrée `jarvis`
> existent (Phase 4). Le découpage interne ci-dessous, lui, reste une cible :
> `audio/`, `stt/`, `tts/` et `llm/` sont pour l'instant les modules plats
> `audio_handler.py`, `stt_module.py`, `tts_module.py`, `llm_module.py` et
> `llm_providers.py`. `tools/`, `policy/` et `speech/` sont en place. Le
> renommage n'apporte rien tant qu'un module tient dans un fichier ; il se
> fera quand l'un d'eux cessera d'y tenir.

```
src/jarvis/
├── __main__.py           point d'entrée, CLI
├── config.py             configuration (pydantic-settings)
├── audio/
│   ├── capture.py        enregistrement, VAD, détection de fin de parole
│   ├── playback.py       lecture interruptible (barge-in)
│   └── devices.py        sélection et diagnostic des périphériques
├── stt/whisper.py        faster-whisper, transcription incrémentale
├── tts/
│   ├── piper.py          synthèse, expose sa fréquence d'échantillonnage
│   └── chunker.py        découpage en phrases pour la synthèse en flux
├── llm/
│   ├── client.py         Ollama, streaming, num_ctx
│   ├── providers.py      fournisseurs ordonnés, sondage, bascule (cf. §6.1)
│   ├── conversation.py   historique structuré + persistance SQLite
│   └── agent.py          la boucle de tool calling
├── tools/
│   ├── registry.py       registre unifié, namespacing, dispatch
│   ├── selection.py      sélection top-k (cf. §4)
│   ├── schema.py         conversion MCP Tool ⇄ schéma de fonction Ollama
│   ├── local/            filesystem (CRUD), web, applications
│   └── mcp/
│       ├── manager.py    enveloppe ClientSessionGroup
│       └── servers.py    lecture de mcp_servers.json
├── policy/
│   ├── engine.py         décision : auto / confirmer / refuser
│   ├── taint.py          suivi du contenu non fiable (cf. §5)
│   └── store.py          approbations persistées (l'actuel whitelist_manager)
└── ui/
    ├── tui.py            interface Rich
    └── confirm.py        surfaces de confirmation par niveau de risque
```

Le point clé : **`action_executor.py` disparaît en tant que point de passage
unique**. Il devient `tools/local/` (des outils comme les autres) plus
`tools/registry.py` (le dispatch) plus `policy/` (la décision). Sans cette
séparation, brancher MCP revient à faire grossir indéfiniment un fichier qui
fait déjà 400 lignes et qui mêle validation, politique et exécution.

---

## 3. La boucle agent

L'implémentation actuelle fait un seul tour d'outils, puis envoie un faux
message utilisateur en anglais pour obtenir une reformulation (BUG-26). La
cible :

```python
async def turn(utterance: str) -> None:
    conversation.add_user(utterance)
    taint = TaintState()

    for _ in range(MAX_TOOL_ITERATIONS):          # borné, défaut 5
        tools = registry.select(utterance, conversation)
        reply = await llm.chat(conversation, tools=tools, stream=True)
        conversation.add_assistant(reply)

        if not reply.tool_calls:
            break

        for call in reply.tool_calls:
            decision = policy.evaluate(call, taint)
            if decision.blocked:
                conversation.add_tool_error(call, decision.reason)
                continue
            if decision.needs_confirmation:
                if not await ui.confirm(decision):
                    conversation.add_tool_error(call, "refusé par l'utilisateur")
                    continue
            result = await registry.call(call)
            taint.observe(call, result)
            conversation.add_tool_result(call, result)

    await speak(reply.text)                        # phrase par phrase
```

Propriétés visées :

- **Bornée** : nombre d'itérations *et* budget temps mural. Un modèle local
  qui boucle sur un appel d'outil ne doit pas pouvoir tourner indéfiniment.
- **Format protocole** : les tours assistant portent `tool_calls`, les
  résultats sont des messages `role: "tool"`. Pas de prose sérialisée.
- **Pas de faux tour utilisateur** : la relance se fait sur l'historique tel
  quel.
- **Streaming** : le texte part vers le TTS phrase par phrase, sans attendre
  la fin de la génération (cf. §6).

---

## 4. Le problème du nombre d'outils

C'est le principal risque technique du projet, et il est propre au
« local-first ».

Un assistant MCP modeste (filesystem + git + sqlite + recherche) expose
facilement **30 à 40 outils**. Deux conséquences :

**Contexte.** Chaque schéma d'outil coûte ~100-300 tokens et il est réinjecté
à *chaque* tour. 35 outils ≈ 5-8k tokens de schéma en permanence. Or Ollama
n'utilise pas la fenêtre native du modèle : `num_ctx` vaut quelques milliers
de tokens par défaut, quel que soit le modèle. Sans réglage explicite, les
schémas d'outils saturent seuls le contexte et la conversation est tronquée
en silence.

→ `num_ctx` doit être exposé en configuration et dimensionné en connaissance
de cause (il coûte de la RAM).

**Précision de sélection.** Un modèle 8B choisit correctement parmi une
poignée d'outils ; au-delà d'une quinzaine, il invente des noms et des
arguments. Aucun réglage de `num_ctx` ne corrige cela.

**Stratégie retenue — sélection top-k par tour :**

1. Namespacer : `filesystem__read_file`, `git__commit` (via le
   `component_name_hook` de `ClientSessionGroup`).
2. Activation par serveur dans la configuration : on ne charge que ce qu'on
   utilise.
3. Si le total ≤ `TOOL_SELECTION_THRESHOLD` (défaut 15) : présenter tous les
   outils. Comportement simple, pas de machinerie inutile.
4. Au-delà : embeddings locaux des descriptions d'outils (calculés une fois au
   démarrage), similarité avec l'énoncé de l'utilisateur, on présente les
   top-k plus un noyau toujours actif (fichiers, web).

Le point 4 introduit `sentence-transformers` — la même dépendance que le RAG
de [`FEATURES_IDEA.md`](FEATURES_IDEA.md) (feature #2), qui la rentabilise.

**Choix de modèle.** La qualité du tool calling local varie beaucoup d'un
modèle à l'autre, et plus vite que les classements publics ne se mettent à
jour. Plutôt que de figer une recommandation ici, le plan prévoit un
harnais d'évaluation (§ Phase 2) : un jeu d'énoncés annotés avec l'outil et
les arguments attendus, exécutable contre n'importe quel modèle Ollama. La
décision se prend alors sur les chiffres de la machine cible, pas sur une
intuition.

---

## 5. Modèle de sécurité

L'ajout de MCP change la nature du problème. Aujourd'hui les trois outils sont
écrits dans le dépôt et audités. Demain, un serveur MCP est un processus tiers
dont **les descriptions d'outils entrent dans le prompt** et dont les
résultats reviennent dans la conversation.

### 5.1 La triade à risque

Jarvis réunit les trois conditions qui rendent l'injection de prompt
exploitable :

| Condition | Chez Jarvis |
|---|---|
| Accès à des données privées | CRUD sur le système de fichiers |
| Ingestion de contenu non fiable | pages web, résultats MCP, descriptions d'outils MCP |
| Canal d'exfiltration | `fetch_web_page` vers une URL arbitraire, serveurs MCP distants |

Un modèle local de 8B résiste *moins bien* à l'injection qu'un modèle
frontière. La défense ne peut donc pas reposer sur le modèle.

### 5.2 Suivi de contamination (`policy/taint.py`)

Mécanisme léger, à l'échelle du tour de conversation :

- Un résultat d'outil est **contaminé** s'il provient du web, d'un serveur MCP
  marqué `open_world_hint`, ou d'un fichier hors d'un répertoire déclaré de
  confiance.
- Une fois le tour contaminé, tout outil qui **écrit, supprime ou émet vers
  l'extérieur** exige une confirmation explicite, *même s'il est dans la
  whitelist*. L'approbation préalable ne couvre que les tours non contaminés.
- La contamination est remise à zéro au tour suivant.

Cela ne bloque pas l'injection, mais coupe la chaîne entre « lire du contenu
hostile » et « agir dessus sans que l'utilisateur le voie ».

### 5.3 Niveaux de confiance par serveur MCP

Déclarés dans `mcp_servers.json`, pas déduits du serveur :

| Niveau | Comportement |
|---|---|
| `trusted` | outils en lecture seule auto-approuvés, écritures confirmées une fois |
| `confirm` | **défaut** — chaque appel confirmé, whitelist possible par outil |
| `readonly` | seuls les outils annotés `read_only_hint` sont exposés |

Les annotations MCP (`read_only_hint`, `destructive_hint`, `idempotent_hint`,
`open_world_hint`) servent à **durcir** la politique, jamais à l'assouplir :
elles sont déclarées par le serveur, donc non fiables. Un outil sans
annotation est traité comme destructif.

### 5.4 Confirmation adaptée au canal vocal

Un assistant vocal ne peut pas faire confirmer une action destructrice à la
voix : la STT se trompe, une conversation ambiante peut déclencher un « oui »,
et l'utilisateur ne voit pas ce qu'il approuve.

| Risque | Surface de confirmation |
|---|---|
| Lecture seule, whitelisté, tour non contaminé | automatique |
| Écriture non destructrice | vocale acceptable |
| Destructive, ou tour contaminé, ou hors sandbox | **TUI/clavier obligatoire**, jamais la voix seule |

La confirmation affiche les **arguments réels**, pas seulement le nom de
l'outil : « supprimer 47 fichiers dans ~/Documents » et non « appeler
`filesystem__delete` ».

### 5.5 Défenses conservées et renforcées

- Sandbox de chemins (déjà solide) étendue aux outils MCP filesystem.
- Motifs refusés (`.ssh`, `.aws`, `.gnupg`, `.env`, `*_history`) évalués après
  la sandbox.
- Garde SSRF sur toute sortie HTTP (schéma, loopback, link-local, RFC 1918).
- Les résultats d'outils sont insérés dans le prompt comme **données
  délimitées**, jamais comme instructions.

---

## 6. Latence

Le pipeline actuel est strictement séquentiel : enregistrer tout → transcrire
tout → générer tout → synthétiser tout → jouer. Ordres de grandeur sur un CPU
récent, pour un énoncé de 5 s :

| Étape | Naïf | Avec streaming |
|---|---|---|
| Détection de fin de parole | ~300 ms | ~300 ms |
| Transcription (whisper base, int8) | 0,5-1,5 s | 0,5-1,5 s |
| Génération LLM complète | 2-6 s | *premier token* 0,5-2 s |
| Synthèse | 0,2-0,4 s / phrase | 0,2-0,4 s (1ʳᵉ phrase) |
| **Premier son entendu** | **3-8 s** | **1,5-3 s** |

Le streaming n'est donc pas un raffinement : c'est ce qui sépare un assistant
d'un traitement par lots. Trois mécanismes :

1. **Génération en flux** — `ollama.chat(stream=True)`.
2. **Découpage en phrases** — dès qu'une phrase complète est disponible, elle
   part au TTS pendant que la suite se génère.
3. **File de lecture** — les segments audio s'enchaînent sans blanc.

Et le corollaire : le **barge-in**. Si l'utilisateur parle pendant que Jarvis
répond, la lecture s'arrête et un nouveau tour commence. Sans cela, une
réponse trop longue ne peut pas être interrompue autrement qu'au clavier.

---

### 6.1 Où tourne le modèle

Un Ollama auto-hébergé sur le LAN et un petit modèle sur cette machine ne sont
pas la même chose, et lequel répond change au cours de la journée. Les
fournisseurs sont donc une **liste ordonnée** : le premier qui répond *et* qui
a son modèle sert le tour.

Trois règles, chacune payée par une erreur qu'elle évite :

1. **Retour automatique au préféré.** Sans cela, la première coupure réseau
   coince la session sur le petit modèle jusqu'au redémarrage. Un sondage est
   refait au bout de `LLM_PROVIDER_RECHECK_SECONDS` tant qu'on n'est pas sur
   le fournisseur préféré ; tant qu'on y est, on ne sonde pas (un aller-retour
   avant chaque réponse, pour rien).
2. **On ne bascule que sur l'injoignable.** Une réponse qui semble mauvaise
   n'est jamais un motif de bascule : changer de modèle en silence au milieu
   d'une conversation est pire que la mauvaise réponse.
3. **Une fois qu'un fragment est prononcé, le tour est engagé.** Si le serveur
   tombe en cours de génération, l'utilisateur a déjà entendu le début ; on
   s'excuse plutôt que de redire la même phrase avec les mots d'un autre
   modèle. Avant le premier fragment, la bascule est invisible.

Un modèle plus petit reçoit une **boîte à outils plus petite** (`max_tools`) :
l'intérêt de basculer sur un 3B est de continuer à fonctionner, pas de garder
une liste d'outils dans laquelle il choisit mal (cf. §4).

---

## 7. Persistance

| Donnée | Emplacement | Raison |
|---|---|---|
| Conversations | SQLite (`~/.local/share/jarvis/history.db`) | reprise de contexte, base du RAG et des analytics |
| Approbations | SQLite, même base | l'actuel `command_whitelist.json`, avec horodatage et portée |
| Configuration serveurs MCP | `mcp_servers.json` | format compatible avec les configurations MCP existantes, donc réutilisable tel quel |
| Liste des fournisseurs LLM | `.env`, ou `llm_providers.json` au-delà de deux | le cas courant (un distant, un local de secours) ne doit pas exiger un fichier |
| Embeddings d'outils | cache disque, invalidé au changement de serveur | évite de recalculer au démarrage |

---

## 8. Ce que l'architecture ne fait pas

À noter explicitement, pour éviter les malentendus :

- **Pas de cloud.** Aucun appel sortant hors des serveurs MCP configurés et
  des URL approuvées. Le mot-clé d'activation devra donc être local
  (`openWakeWord` plutôt que Porcupine, qui exige une clé d'API).
- **Pas de multi-utilisateur** en première intention. La reconnaissance de
  locuteur (feature #9) suppose un modèle de permissions par utilisateur, qui
  est un projet en soi.
- **Pas de serveur MCP côté Jarvis** dans un premier temps. Exposer Jarvis
  *comme* serveur MCP est intéressant (feature #13) mais orthogonal, et ne
  doit pas retarder le rôle de client.
