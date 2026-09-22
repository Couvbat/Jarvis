# Jarvis — Plan d'implémentation

**Cible** : assistant vocal 100 % local, tool calling, CRUD système de
fichiers, connexion MCP à des services externes.
Architecture visée : [`ARCHITECTURE.md`](ARCHITECTURE.md).
État actuel : [`AUDIT.md`](AUDIT.md).
Campagne manuelle : [`TESTING.md`](TESTING.md).

> **Phases 0 à 4 livrées**, plus les fournisseurs LLM ordonnés (3.5), le mot
> d'activation (5.1), la recherche documentaire (5.2) et les rappels (5.3).
> 1209 tests passants, 0 `xfail`, 97 % de couverture. Le reste de la Phase 5
> est un choix de fonctionnalités, pas une dette.

**Règle de sortie de chaque tâche** : le ou les tests `xfail(strict)`
correspondants passent au vert. Un `xfail` strict qui réussit fait échouer
`pytest`, donc le correctif et le retrait de son marqueur avancent ensemble
par construction.

---

## ⚠️ Séquencement : Phase 0 et Phase 1 ne se séparent pas

Aujourd'hui, **aucun outil ne s'exécute jamais** : BUG-08 fait échouer le
chemin nominal d'appel d'outil. Les failles de sécurité de l'audit (lectures
non confirmées, absence de garde SSRF) ne sont donc pas exploitables — elles
sont simplement inatteignables.

La Phase 0 les rend atteignables. **Ne pas livrer la Phase 0 sans la
Phase 1.** Si le temps manque, il vaut mieux un Jarvis cassé qu'un Jarvis
fonctionnel qui lit `~/.ssh/id_rsa` sans rien demander.

---

## Phase 0 — Rendre le produit fonctionnel ✅ livrée

Prérequis absolu de tout le reste. Détail des correctifs dans
[`AUDIT.md`](AUDIT.md).

Les 13 marqueurs `xfail` de cette phase ont été retirés : 379 tests passants,
24 `xfail` restants, 96 % de couverture. Vérifié aussi par un run réel du
point d'entrée `main.py --text`, qui produit l'historique au format protocole
`[system, user, assistant(tool_calls), tool]` et crée le fichier demandé.

En plus des 5 tâches prévues, la refactorisation de l'historique a rendu
gratuits BUG-07 (`max_history=0`) et BUG-11 (prompt système dupliqué), tous
deux planifiés en Phase 2 — ils sont faits.

**Écart assumé par rapport au plan initial** : 0.1 devait se contenter de
passer `CHUNK_SIZE` à 320 avec une validation au démarrage. La taille de
trame du VAD a plutôt été découplée de la taille de bloc de capture, si bien
que la détection de silence fonctionne à n'importe quel `CHUNK_SIZE`.
`CHUNK_SIZE` est une option documentée et exposée à l'utilisateur : la valeur
avec laquelle le projet était livré serait sinon devenue une erreur fatale.

| Tâche | Bugs | Charge |
|---|---|---|
| VAD : trame légale (`CHUNK_SIZE=320`) + validation au démarrage | BUG-18 | 30 min |
| Audio mono 1-D vers Whisper ; canal 0 vers le VAD | BUG-21, BUG-19 | 20 min |
| Protocole d'appel d'outils Ollama (cf. ci-dessous) | BUG-08, BUG-09, BUG-26 | 3 h |
| Découverte Piper + fréquence d'échantillonnage propagée | BUG-15, BUG-16, BUG-17, BUG-27 | 1 h |
| Commande de sortie : comparaison exacte, pas en sous-chaîne | BUG-22 | 30 min |

Le correctif structurant est le troisième. Il conditionne toute la suite, car
c'est lui qui donne à l'historique la forme que MCP et la persistance
attendent :

1. `ConversationHistory` porte des champs structurés (`tool_calls`,
   `tool_name`), pas seulement rôle + contenu.
2. Les tours assistant sont stockés au format protocole
   (`{"role": "assistant", "tool_calls": [...]}`), les résultats en
   `{"role": "tool", "name": ..., "content": ...}`. Normalisation des modèles
   Pydantic via `model_dump()` — plus de `json.dumps`.
3. Remplacement du faux tour utilisateur « Please provide a natural
   response… » par une relance sur l'historique existant.
4. Épingler `ollama>=0.4,<1.0`.

*Tests concernés* : `test_tool_call_without_content_survives_history_recording`,
`test_tool_calls_are_replayed_in_the_protocol_shape`,
`test_follow_up_after_tools_is_not_a_user_turn`,
`test_file_creation_works_with_a_modern_ollama_client`, plus les 8 autres
`xfail` listés dans l'audit pour cette phase.

---

## Phase 1 — Registre d'outils, CRUD complet, moteur de politique ✅ livrée

Les 10 marqueurs `xfail` de cette phase ont été retirés : 604 tests passants,
14 `xfail` restants, 95 % de couverture. `action_executor.py` et
`whitelist_manager.py` sont supprimés.

**Trouvé par le run réel, pas par les tests unitaires** : le moteur demandait
confirmation *avant* toute vérification du sandbox, donc lire `/etc/passwd`
interrogeait l'utilisateur pour n'être refusé qu'ensuite. `ToolSpec.precheck`
corrige l'ordre — une invite dépense l'attention de l'utilisateur, et la
dépenser sur une action impossible lui apprend à valider sans lire.

**Durcissements non prévus au plan, décidés en cours de route** :

- `ALLOWED_DIRECTORIES` passe de `/home,/tmp` à `~/Documents,~/Downloads,/tmp`.
- `COMMAND_WHITELIST` perd `ls`, `cat`, `mkdir`, `touch` et `rm` : un outil CLI
  whitelisté atteint tout le disque, car il ne passe jamais par le sandbox de
  chemins. `rm` dans la whitelist par défaut contournait entièrement la
  protection des fichiers.

---

## Phase 1 — détail (livrée)

C'est la phase qui prépare MCP. Elle démonte `action_executor.py` en trois
responsabilités séparées, et elle traite la sécurité **dans** cette
refactorisation plutôt qu'en rustines qu'il faudrait refaire ensuite.

### 1.1 — Registre d'outils · 1 j
`tools/registry.py`, `tools/schema.py`, `tools/local/`

- Un outil est déclaré par sa métadonnée (nom, description, schéma JSON
  d'entrée, annotations de risque) et sa fonction d'exécution. Fin de la liste
  `TOOLS` codée en dur dans `llm_module.py` et du `if/elif` de dispatch dans
  `action_executor.py`.
- `registry.describe()` produit les schémas de fonction pour Ollama ;
  `registry.call(name, args)` dispatche. Même chemin pour un outil local et,
  demain, pour un outil MCP.
- Namespacing dès maintenant (`fs__read_file`), pour ne pas avoir à migrer les
  noms quand les serveurs MCP arriveront.
- Conversion `mcp.types.Tool` ⇄ schéma Ollama isolée dans `schema.py`
  (`input_schema` → `function.parameters`), testable sans serveur MCP.

Le test `TestToolSchemaAgreement` de la suite actuelle vérifie déjà que tout
outil déclaré est dispatchable — il devient la garantie de non-régression de
cette refactorisation.

### 1.2 — CRUD complet sur le système de fichiers · 1 j
`tools/local/filesystem.py`

L'accès CRUD est un objectif affiché ; aujourd'hui il manque le U et une
partie du D.

| Opération | État | À faire |
|---|---|---|
| Create | `create_file`, `create_directory` | — |
| Read | `read_file`, `list_directory` | pagination, lecture par plage de lignes |
| **Update** | **absent** | `write_file`, `append_file`, `replace_in_file` |
| Delete | `delete_file` | `delete_directory` (récursif, confirmation renforcée) |
| — | — | `move`, `copy`, `stat`, `search` (glob + contenu) |

Points d'attention : lecture par plage de lignes (un fichier de 10 000 lignes
ne rentre pas dans le contexte d'un modèle local), détection binaire avant
lecture, et `replace_in_file` sur correspondance unique — un remplacement
ambigu doit échouer plutôt que de modifier au hasard.

*Tests* : `test_delete_directory_operation_should_exist`,
`test_append_to_file_should_exist` (BUG-03, BUG-05).

### 1.3 — Moteur de politique · 1,5 j
`policy/engine.py`, `policy/taint.py`, `policy/store.py`

Le cœur du modèle de sécurité décrit dans [`ARCHITECTURE.md`](ARCHITECTURE.md) §5.

- **Décision** : `evaluate(tool, args, taint) → auto | confirm(surface) | deny`,
  à partir du niveau de risque de l'outil, de la sandbox, des approbations
  enregistrées et de l'état de contamination.
- **Toutes** les opérations fichier passent par là, y compris les lectures
  (BUG-01). Catégories d'approbation distinctes par opération : approuver une
  lecture n'approuve pas une suppression.
- **Contamination** : un tour ayant ingéré du contenu non fiable exige
  confirmation pour toute écriture ou sortie réseau, whitelist ou non.
- **Confirmation graduée** : la voix ne suffit jamais pour une action
  destructive (§5.4). La confirmation affiche les arguments réels.
- **Motifs refusés** (`.ssh`, `.aws`, `.gnupg`, `.env`, `*_history`) évalués
  après la sandbox.
- Migration de `command_whitelist.json` vers SQLite, avec portée et horodatage.

*Tests* : les trois `test_*_should_require_confirmation` (BUG-01).

### 1.4 — Garde réseau · 0,5 j
`tools/local/web.py`

Schéma restreint à `http`/`https` ; refus de loopback, link-local et RFC 1918
sauf autorisation explicite ; `stream=True` avec coupure à `MAX_FETCH_BYTES` ;
refus des redirections changeant d'hôte vers une plage interdite. Whitelist de
commandes comparée sur le nom de base résolu, et `shlex.split()` pour les
arguments.

*Tests* : `test_non_http_scheme_should_be_rejected`,
`test_loopback_url_should_be_rejected`,
`test_oversized_response_should_be_refused_before_parsing`,
`test_command_string_containing_arguments_should_run`,
`test_absolute_path_to_a_listed_command_should_be_normalised`
(BUG-04, BUG-02, BUG-06).

---

## Phase 2 — Passage en asynchrone et client MCP ✅ livrée

771 tests passants, 14 `xfail`, 95 % de couverture. Vérifié par un vrai
serveur MCP en sous-processus traversant toute la boucle.

**Trois choses trouvées par les runs réels, pas par les tests unitaires :**

1. Sous `trust: trusted`, les outils lecture seule demandaient quand même
   confirmation. Écrire `"trust": "trusted"` dans la configuration *est*
   l'approbation → `ToolSpec.preapproved`.
2. Le premier appel MCP contaminait le tour, donc tous les suivants étaient
   escaladés au clavier : un workflow multi-étapes devenait un mur d'invites.
   La règle de taint demandait *si* le tour était sale, pas *où* le contenu
   irait. Les résultats portent désormais une **origine**, et la sortie n'est
   escaladée que vers une origine nouvelle pour ce tour.
3. Un serveur bloqué au démarrage attendait les 5 s de grâce avant d'être
   abandonné, retardant tous les suivants.

**Écart assumé** : le plan prévoyait `sentence-transformers` pour la sélection
d'outils. La sélection est lexicale, avec une table de synonymes FR↔EN — le
rappel mesuré est de 100 % au `top_k` par défaut. Un backend d'embeddings
multilingue coûte une pile d'apprentissage profond pour classer quarante
chaînes courtes ; il doit s'adopter sur les chiffres de `tests/eval`, pas par
principe. La table de synonymes a un plafond connu et il est documenté.

**BUG-34** (nouveau, trouvé en Phase 2) : `OLLAMA_HOST` était lu dans les
réglages puis jamais utilisé — le `ollama.chat()` de module construit son
propre client depuis l'environnement du processus. Pointer `.env` vers une
autre machine ne faisait rien. Corrigé par `AsyncClient(host=...)`.

---

## Phase 2 — détail (livrée)

### 2.1 — Passage du pipeline en asynchrone · 1,5 j

**À traiter avant MCP, pas pendant.** Le SDK MCP est entièrement asynchrone
(`ClientSession`, `stdio_client`, `ClientSessionGroup` sont des gestionnaires
de contexte async) ; le code actuel est entièrement synchrone. Deux options :

| Option | Verdict |
|---|---|
| Boucle d'événements dans un thread, ponts `run_coroutine_threadsafe` | Rejetée — la gestion du cycle de vie des sessions MCP devient fragile, et le streaming et le barge-in de la Phase 3 exigeront l'asynchrone de toute façon |
| **Boucle agent et couche outils en `async`** | **Retenue** — l'audio (I/O bloquantes) reste dans des threads via `asyncio.to_thread` |

Faire ce passage sur la couche outils *avant* d'y brancher MCP évite de le
faire deux fois.

### 2.2 — Gestionnaire de serveurs MCP · 1,5 j
`tools/mcp/manager.py`, `tools/mcp/servers.py`

- S'appuyer sur `mcp.client.session_group.ClientSessionGroup`, qui agrège déjà
  plusieurs serveurs et expose `tools`, `resources`, `prompts` et `call_tool`.
  Son `component_name_hook` fournit le namespacing (§4) — ne pas le
  réimplémenter.
- Transports : `StdioServerParameters` pour les serveurs locaux (le cas
  nominal en local-first), `StreamableHttpParameters` et `SseServerParameters`
  pour les distants.
- Configuration dans `mcp_servers.json`, au format des configurations MCP
  existantes, pour que l'utilisateur réutilise ses serveurs sans les
  réécrire — plus les champs propres à Jarvis : `enabled`, `trust`
  (`trusted` | `confirm` | `readonly`).
- Cycle de vie robuste : un serveur qui ne démarre pas, qui meurt en cours ou
  qui ne répond pas ne doit pas empêcher Jarvis de fonctionner avec les
  autres. Démarrage en parallèle avec délai d'attente, journalisation claire,
  reconnexion à la demande.
- `mcp>=2.2,<3` dans les dépendances (aujourd'hui `pyproject.toml`).

### 2.3 — Intégration au registre et à la politique · 1 j

- Les outils MCP entrent dans le même registre que les outils locaux ; la
  boucle agent ne fait pas la différence.
- Les annotations MCP (`read_only_hint`, `destructive_hint`,
  `idempotent_hint`, `open_world_hint`) alimentent le moteur de politique en
  **durcissement uniquement** : un outil non annoté est traité comme
  destructif. Une annotation ne peut jamais abaisser le niveau de
  confirmation.
- Les descriptions d'outils MCP entrent dans le prompt : elles sont du contenu
  tiers. Les tronquer, les délimiter, et exiger que l'utilisateur voie les
  outils exposés par un serveur la première fois qu'il l'active.

### 2.4 — Sélection d'outils top-k · 1 j
`tools/selection.py`

Le mécanisme décrit en [`ARCHITECTURE.md`](ARCHITECTURE.md) §4 : sous le seuil,
tous les outils ; au-delà, embeddings locaux et top-k plus noyau permanent.
Exposer `num_ctx` dans les options Ollama et le dimensionner en fonction du
volume de schémas.

*À vérifier sur la machine cible* : le seuil de 15 outils est une hypothèse
de départ, à confirmer avec le harnais d'évaluation ci-dessous.

### 2.5 — Harnais d'évaluation du tool calling · 0,5 j
`tests/eval/`

Un jeu d'énoncés annotés (« crée un fichier notes.txt dans Documents » →
`fs__create_file`, `path=~/Documents/notes.txt`), exécutable contre un modèle
Ollama réel, mesurant le bon choix d'outil et la justesse des arguments en
fonction du nombre d'outils exposés.

Marqué `@pytest.mark.slow` et exclu du CI (il exige un serveur Ollama). C'est
ce qui permet de choisir le modèle et de régler le seuil sur des chiffres
plutôt que sur une intuition.

---

## Phase 3 — Latence, persistance, usage quotidien ✅ livrée

953 tests passants, **0 `xfail`**, 96 % de couverture. Les 34 anomalies de
l'audit sont closes.

**Trouvé par les tests, pas par le raisonnement** : router la parole dans le
streaming rendait un Ollama mort *silencieux* — le chemin d'erreur renvoyait
son message sans jamais appeler `on_text`.

**Limite assumée et documentée** : le barge-in est désactivé par défaut. Un
micro dans la même pièce qu'un haut-parleur entend le haut-parleur ; sans
annulation d'écho, Jarvis se coupe lui-même. Le mécanisme est complet et
testé (plancher de bruit calibré sur l'écho, parole soutenue et
ininterrompue, mots conservés pour le tour suivant) et fonctionne au casque.

**Sommes de contrôle réelles** : les six empreintes SHA-256 de `setup_piper.py`
ont été obtenues en téléchargeant et hachant les fichiers publiés. Livrer le
mécanisme avec des valeurs fictives aurait été pire que de ne rien livrer.

---

## Phase 3 — détail (livrée)

C'est la phase qui fait passer de « ça marche » à « je m'en sers tous les
jours ».

| Tâche | Détail | Charge |
|---|---|---|
| Génération LLM en flux | `ollama.chat(stream=True)` de bout en bout | 0,5 j |
| Découpage en phrases + file TTS | Le premier son part avant la fin de la génération : ~3-8 s → ~1,5-3 s | 1 j |
| Barge-in | Parler pendant la réponse l'interrompt et ouvre un nouveau tour | 1 j |
| Persistance des conversations | SQLite ; quasi gratuit une fois la Phase 0.3 faite | 0,5 j |
| Contrôle de disponibilité Ollama + dégradation | BUG-10 | 0,5 j |
| Correctifs TUI | BUG-29 (les 2 messages les plus récents sont coupés — le plus visible à l'usage), BUG-28, BUG-30 | 0,5 j |

Le reste des écarts de l'audit (BUG-07, 11, 12, 13, 14, 20, 23, 24, 25, 33) se
traite au fil de ces phases ; ce sont des correctifs de 15 min à 2 h chacun,
tous couverts par un test `xfail`.

---

## Phase 3.5 — Fournisseurs LLM ordonnés ✅ livrée (hors plan)

Demandée en cours de route : un Ollama auto-hébergé sur le NAS, avec repli sur
un petit modèle local quand il n'est pas joignable. 1022 tests passants, 100 %
de couverture sur `llm_providers.py` et `llm_module.py`.

| Tâche | Détail |
|---|---|
| `llm_providers.py` | liste ordonnée, sondage (`list`) avec délai, table de santé par fournisseur |
| Bascule dans `LLMModule` | les fournisseurs sont essayés dans l'ordre pour chaque tour ; `_stream_turn` remplit `parts` au fil de l'eau pour que l'appelant sache ce qui a déjà été prononcé |
| Retour au préféré | nouveau sondage au bout de `LLM_PROVIDER_RECHECK_SECONDS` tant qu'on est sur un repli |
| Boîte à outils réduite | `max_tools` par fournisseur : un 3B choisit mal dans la liste d'un 32B |
| Rapport au démarrage | `main.py` annonce le fournisseur qui répond *et* l'état des autres |
| Configuration | `.env` pour le cas à deux ; `llm_providers.json` au-delà |

**Décisions prises et payées par un test** :

- **On ne bascule que sur l'injoignable ou le modèle absent.** Une réponse qui
  semble mauvaise n'est pas une condition de bascule.
- **Une fois un fragment prononcé, le tour est engagé** sur ce fournisseur :
  redire la même phrase avec les mots d'un autre modèle serait pire que de
  s'excuser. Ce qui a été prononcé reste dans l'historique, avec l'excuse
  attachée, pour que la trace corresponde à ce que l'utilisateur a entendu.
- **Une liste de modèles vide n'est pas une preuve d'absence** : on tente le
  fournisseur plutôt que de basculer pour rien.
- **L'annonce de bascule est dissociée de `active`.** Premier jet : `refresh()`
  comparait avec `self.active`, que `report_failure()` venait justement
  d'effacer — le seul message qui compte, « le modèle a changé », était
  précisément celui qui ne s'affichait jamais. Trouvé en cherchant pourquoi
  une ligne restait non couverte.

**Trouvé par le smoke test, pas par les tests unitaires** : rien. Le scénario
complet (NAS absent au démarrage, retour du NAS entre deux tours) passe du
premier coup à travers le vrai `main.py --text`.

---

## Phase 4 — Qualité et packaging ✅ livrée

- **Nettoyage lint** en deux commits, pas un seul. Élargir le filtre a sorti
  518 remarques, dont quatre qui n'étaient pas du style : une clé `"site"` en
  double dans `TERM_ALIASES` — la seconde remplaçait silencieusement la
  première, et `"url"` cessait d'être une expansion de `"site"` —, deux levées
  d'exception sans `from e`, et une affectation morte. Corrigées d'abord, pour
  qu'un diff mécanique de 837 lignes ne les enterre pas. Puis `ruff --fix` sur
  `["E","F","W","I","UP","B"]` : 155 lignes vides avec espaces, 246
  annotations passées en PEP 585/604, 54 imports `typing` obsolètes, 13
  imports morts, 10 blocs d'imports triés.
- **`ruff format` écarté**, et `ruff.toml` dit pourquoi : 54 fichiers et 2500
  lignes réécrites, pour un résultat moins lisible que ce qu'il remplace (les
  tables d'alias de `text_utils.py` à une entrée par ligne, les littéraux
  d'appels d'outils des tests profondément imbriqués). Les défauts étaient
  chez le linter ; le formateur n'aurait fait que les déplacer.
- **Packaging** : `pyproject.toml`, modules sous `src/jarvis/`, deux points
  d'entrée (`jarvis`, `jarvis-setup-piper`), bornes de version majeure.
  `requirements.txt` et `requirements-dev.txt` supprimés ;
  `requirements-test.txt` reste et explique dans son en-tête pourquoi il ne
  peut pas être un extra du paquet.
- **Découpage interne inchangé.** Le §2 d'`ARCHITECTURE.md` décrit `audio/`,
  `stt/`, `tts/`, `llm/` ; ce sont toujours des modules plats. Renommer
  n'apporte rien tant qu'un module tient dans un fichier, et le document dit
  désormais lequel existe et lequel reste une cible, plutôt que de décrire une
  arborescence absente.
- **CI** : un travail `package` en plus. Les tests peuvent retomber sur
  l'arborescence source, donc une roue à laquelle il manque un sous-paquet les
  passerait quand même ; celui-là construit la roue, l'installe sans
  dépendances et importe au travers.
- **`LICENSE`** : MIT, que le README annonçait depuis le premier commit.
- **`TESTING.md`** : la campagne manuelle, en dix sections. Écrite contre le
  code, pas en général : les noms d'outils, les surfaces de confirmation et
  les scénarios de contamination y sont ceux que le moteur applique vraiment.

**Vérifié au-delà de la suite**, parce que les annotations modernisées sont
évaluées à l'exécution par pydantic et par les dataclasses : tous les modules
compilent, `get_type_hints` résout sur `ToolSpec`, `ToolResult`,
`ProviderConfig` et `ProviderState`, la roue embarque les 8 sous-paquets, et
un cinquième smoke test résout `jarvis` depuis les métadonnées de point
d'entrée et mène un tour complet au travers.

---

## Phase 5 — Fonctionnalités

Relecture de [`FEATURES_IDEA.md`](FEATURES_IDEA.md) à la lumière de l'objectif.
MCP change le classement de façon importante.

### Absorbées par MCP — ne plus les développer en code propre

| Feature | Devient |
|---|---|
| #4 Système de plugins | **Superflu.** MCP *est* le système d'extension. Développer les deux serait faire deux fois le même travail avec deux modèles de sécurité à maintenir. |
| #7 Email et agenda | Serveurs MCP existants. Zéro code dans Jarvis. |
| #20 Outils de dev (git, docker, bases de données) | Serveurs MCP existants. |
| #5 Supervision système | Serveur MCP, ou petit outil local si la latence compte. |

C'est le principal gain de l'orientation MCP : quatre fonctionnalités estimées
à 27-36 h dans `FEATURES_IDEA.md` deviennent des lignes de configuration.

### Restent du code propre à Jarvis

| Feature | Position | Remarque |
|---|---|---|
| #3 Persistance des conversations | **Phase 3** | Prérequis du RAG, des analytics et du multi-utilisateur |
| #6 Planification et rappels | ✅ **livrée (5.3)** | Ni `schedule` ni APScheduler : la livraison a lieu entre les tours de la boucle existante, et le modèle fait la conversion des dates à la place d'une bibliothèque. |
| #16 Meilleure récupération sur erreur | Réparti sur 0→3 | Largement traité en chemin |
| #1 Mot-clé d'activation | ✅ **livrée (5.1)** | openWakeWord, qui tourne en local et livre un modèle `hey_jarvis` pré-entraîné. Porcupine écarté comme prévu : une clé d'API Picovoice est exactement ce que « 100 % local » exclut. |
| #2 RAG documentaire | ✅ **livrée (5.2)** | Le partage d'embeddings avec la sélection d'outils n'a jamais eu lieu : 2.4 a finalement été lexicale. Les embeddings viennent donc d'Ollama (`nomic-embed-text`), ce qui coûte zéro dépendance Python là où `sentence-transformers` aurait coûté torch. |
| #11 Vision (LLaVA) | Plus tard | Ollama gère les modèles multimodaux ; le coût est surtout en RAM |
| #9 Multi-utilisateur | Plus tard | Suppose un modèle de permissions par utilisateur — un projet en soi |
| #13 API / WebSocket | Plus tard | Devient « Jarvis exposé *comme* serveur MCP », plus intéressant que REST, mais orthogonal au rôle de client |

---

## Phase 5.1 — Mot d'activation ✅ livrée

Le premier élément de la Phase 5 dont les prérequis étaient tenus : un VAD qui
marche (0.1) et une boucle audio continue (3.2).

| Tâche | Détail |
|---|---|
| `speech/wake.py` | détecteur openWakeWord, trames de 1280 échantillons, import paresseux |
| `AudioHandler.wait_for_wake` | écoute sans enregistrer, puis rend l'audio qui suit la phrase |
| Fenêtre de relance | après une réponse, `WAKE_WORD_FOLLOW_UP` secondes sans avoir à répéter la phrase |
| Dépendance optionnelle | extra `[wakeword]` : onnxruntime et tflite-runtime n'ont pas de roue partout, et taper ses commandes reste légitime |

**Décisions payées par un test** :

- **La commande dite dans la même respiration est conservée.** On dit « hey
  Jarvis quelle heure est-il », pas « hey Jarvis », pause, « quelle heure
  est-il ». L'audio qui suit la phrase est passé en `prefix` au même mécanisme
  que le barge-in utilise pour les mots prononcés avant qu'il ne réagisse.
- **Une relance n'exige pas la phrase.** La redire à chaque tour d'un échange
  est ce que les gens cessent de faire.
- **Une interruption n'exige jamais la phrase.** Le barge-in *est* déjà
  l'utilisateur qui parle.
- **Paquet ou modèles absents : Jarvis démarre quand même** et écoute comme
  avant, en le disant une fois. Refuser de démarrer parce que le mains-libres
  est indisponible serait pire que de ne pas l'avoir.

**Trouvé en y réfléchissant, pas par les tests** : l'attente tourne dans un
fil que Ctrl-C n'atteint pas, et l'interpréteur joint ce fil en sortant — une
session réveillée ne se terminait donc jamais. Vérifié en envoyant un SIGINT à
un Jarvis en attente : le processus devait être tué. Corrigé par un drapeau
d'arrêt posé par `aclose()` et relu par l'écoute, avec un test de
non-régression.

**Non vérifiable ici** : la détection réelle dans une pièce, les faux positifs
sur une télévision allumée, la portée. Section 9 de [`TESTING.md`](TESTING.md).

---

## Phase 5.2 — Recherche documentaire ✅ livrée

| Tâche | Détail |
|---|---|
| `rag/store.py` | SQLite + cosinus en force brute sur numpy. 20 000 × 768 float32 = 61 Mo et quelques millisecondes : chromadb n'achèterait rien et coûterait une dépendance, un démon et un second endroit où les données rancissent. |
| `rag/embeddings.py` | Ollama, via le même `ProviderPool` que le chat — donc calculés sur le NAS quand le NAS est là, et repli identique |
| `rag/indexer.py` | découpe sur les paragraphes, puis les phrases, puis les caractères ; incrémental par mtime et taille |
| `jarvis-index` | troisième point d'entrée, avec `--status`, `--force` |
| `tools/local/documents.py` | `docs__search`, enregistré **seulement** s'il y a un index à interroger |

**Décisions payées par un test** :

- **Embeddings par Ollama, pas `sentence-transformers`.** Le plan tablait sur
  un partage avec la sélection d'outils ; celle-ci étant restée lexicale, il
  n'y avait rien à partager. Ollama tourne déjà : zéro dépendance ajoutée, et
  les embeddings suivent le fournisseur qui répond.
- **La recherche est un outil que le modèle appelle**, pas du contexte injecté
  à chaque tour : « quelle heure est-il » ne doit pas traîner vos notes dans
  l'invite.
- **Le modèle d'embedding est enregistré avec l'index.** Deux modèles donnent
  des vecteurs incomparables et l'échec est *silencieux* : la recherche
  continue de marcher et renvoie des absurdités assurées. Refusé, donc, pas
  moyenné.
- **Le contenu retrouvé est non fiable**, ce que `fs__read` n'est pas : là,
  c'est l'utilisateur qui a nommé le fichier ; ici, c'est un score de
  similarité qui l'a choisi. L'`origin` étant l'index et non chaque fichier,
  chercher deux fois ne réescalade pas — c'est exactement ce pour quoi la
  règle d'origine existe.
- **Le bac à sable s'applique à l'indexation.** Indexer un fichier, c'est
  mettre son contenu à une similarité de l'invite.

**Trouvé en cherchant pourquoi une branche restait non couverte** : les k plus
proches renvoient *toujours* quelque chose, donc la branche « rien ne
correspond » était morte — et une question dont les notes ne parlent pas
revenait avec les cinq passages les moins hors-sujet, que le modèle résumait
ensuite avec aplomb. D'où `RAG_MIN_SIMILARITY`, désactivé par défaut parce que
le bon seuil dépend du modèle, avec les similarités affichées pour le calibrer.

**Trouvé par le smoke test** : `jarvis --text` sur une entrée épuisée
(`echo … | jarvis --text`) finissait en « Fatal error » et code 1, et un EOF
sur une demande de confirmation faisait échouer le tour au lieu de refuser.
Corrigés tous les deux.

**Non vérifiable ici** : la qualité de rappel sur de vrais documents avec un
vrai modèle d'embedding. Section 9 de [`TESTING.md`](TESTING.md).

---

## Phase 5.3 — Rappels ✅ livrée

| Tâche | Détail |
|---|---|
| `reminders.py` | SQLite ; `pending`, `due`, annulation, plafond d'ancienneté |
| `tools/local/schedule.py` | `remind__set`, `remind__list`, `remind__cancel` |
| Horloge dans l'invite | rafraîchie à chaque tour — une constante serait fausse en quelques heures |
| Livraison | entre les tours, et en interrompant l'attente du mot d'activation |

**Décisions payées par un test** :

- **Pas de bibliothèque d'analyse de dates.** `dateparser` existe pour
  transformer « demain à 9 h » en horodatage, mais il y a déjà dans la boucle
  un modèle de langue dont c'est tout le métier. L'outil reçoit donc un
  horodatage et le **vérifie** : un rappel dans le passé ou à un an est rendu
  au modèle pour qu'il recommence, plutôt que posé au mauvais moment.
- **Les rappels se déclenchent entre les tours, jamais pendant.** Parler
  par-dessus un enregistrement met la voix de Jarvis dans le micro — c'est le
  problème pour lequel le barge-in existe et reste désactivé par défaut.
  Quelques secondes de retard que personne ne remarque, contre un assistant
  qui vous coupe la parole.
- **Trois niveaux de risque différents** : poser un rappel est un `WRITE`
  (un engagement pris en votre nom, mais annulable), lister est pré-approuvé
  (ce sont vos propres rappels), annuler est `DESTRUCTIVE` — une promesse
  discrètement abandonnée, et on ne s'en aperçoit qu'à l'heure où elle aurait
  dû être tenue.
- **Plafond d'ancienneté d'une semaine.** Se réveiller avec un mois d'arriéré
  serait pire que de perdre le rappel ; il reste visible dans la liste.

**Trouvé par une mutation** : en retirant `mark_fired`, la suite ne partait
pas en échec mais en **boucle infinie** — la boucle repart écouter dès qu'un
rappel est dû, donc un marquage qui échoue (base verrouillée, disque plein)
aurait fait répéter le rappel jusqu'à l'arrêt. Corrigé par un jeu en mémoire
des rappels déjà prononcés, avec son test.

**Deux tests qui ne mesuraient rien**, repérés en mutant : l'un vérifiait le
prédicat d'arrêt *après* l'arrêt (donc toujours vrai), l'autre plaçait un
rappel déjà dû avant une attente qui ne commence qu'une fois les rappels
livrés. Le cas réel — un rappel qui échoit *pendant* l'attente — est
maintenant modélisé par un crochet dans la doublure audio.

**Non vérifiable ici** : si un modèle local calcule juste « demain à 9 h ».
Section 9 de [`TESTING.md`](TESTING.md), et c'est le vrai risque de cette
fonctionnalité.

---

## Charge totale

| Phase | Charge |
|---|---|
| 0 — Fonctionnel | 1 j |
| 1 — Registre, CRUD, politique | 4 j |
| 2 — Async et MCP | 5 j |
| 3 — Latence et usage quotidien | 4 j |
| 3.5 — Fournisseurs LLM ordonnés (hors plan) | 0,5 j |
| 4 — Qualité et packaging | 2 j |
| **Jusqu'à l'objectif affiché** | **~16 j — livrée** |
| 5.1 — Mot d'activation | 0,5 j |
| 5.2 — Recherche documentaire | 1 j |
| 5.3 — Rappels | 0,5 j |

Phase 5 selon les priorités, la majeure partie étant devenue de la
configuration.

---

## Suite de tests

### Exécution

```bash
pip install -r requirements-test.txt   # pas de PortAudio, pas de modèle, pas d'Ollama
pytest                                 # ~1,2 s
pytest --cov --cov-report=term-missing
ruff check .
```

### État actuel

1022 tests passants, **0 `xfail`**, 96 % de couverture.

Les fichiers ajoutés depuis l'audit : `test_policy_paths.py`,
`test_policy_engine.py`, `test_tool_schema.py`, `test_tool_registry.py`,
`test_fs_tools.py`, `test_web_tools.py`, `test_app_tools.py`,
`test_selection.py`, `test_mcp_servers.py`, `test_mcp_manager.py`,
`test_mcp_adapter.py`, `test_builtin_assembly.py`, `test_speech.py`,
`test_barge_in.py`, `test_conversation_store.py`, `test_llm_providers.py`,
et `tests/eval/`.

### État au moment de l'audit

344 tests, 306 passants et 38 `xfail(strict)`, 96 % de couverture.

```
tests/
├── _stubs.py                          doublures sounddevice / webrtcvad /
│                                      faster-whisper / ollama
├── conftest.py                        isolation d'environnement, fixtures
├── test_config.py                     11 tests
├── test_whitelist_manager.py          17
├── test_action_executor_paths.py      15  sandbox, traversée, symlinks
├── test_action_executor_files.py      35  CRUD, confirmations, whitelist
├── test_action_executor_web.py        15  extraction, erreurs, SSRF
├── test_action_executor_apps.py       23  whitelist, GUI vs CLI, sorties
├── test_action_executor_dispatch.py   12  routage, appels malformés
├── test_llm_module.py                 31  historique, protocole Ollama
├── test_stt_module.py                 22  chargement, normalisation, langues
├── test_tts_module.py                 24  binaire, modèle, synthèse
├── test_audio_handler.py              27  capture, VAD, lecture, fichiers
├── test_tui.py                        35  état, rendu, confirmations
├── test_main.py                       54  orchestration, modes, CLI, TUI
├── test_setup_piper.py                11  plateformes, extraction, intégrité
└── test_integration.py                12  flux bout-en-bout
```

### Extensions prévues par phase

| Phase | Ajouts |
|---|---|
| 1 | `test_registry.py` (dispatch, namespacing, collisions), `test_policy_engine.py` (matrice de décision), `test_taint.py`, CRUD complet dans `test_filesystem.py` |
| 2 | `test_mcp_schema.py` (conversion, sans serveur), `test_mcp_manager.py` (serveur MCP factice en mémoire — le SDK fournit `mcp.client._memory` pour cela), `test_tool_selection.py`, `tests/eval/` (hors CI) |
| 3 | `test_streaming.py` (découpage en phrases, ordre de la file), `test_barge_in.py`, `test_persistence.py` |

### Principes retenus

- **Les doublures refusent ce que refusent les vraies bibliothèques.** La
  doublure `webrtcvad` rejette les trames de 64 ms, la doublure `ollama`
  renvoie des objets non sérialisables en JSON. Une doublure permissive aurait
  masqué BUG-18 et BUG-08, les deux anomalies les plus graves du dépôt. Le même
  principe s'appliquera au serveur MCP factice : il doit respecter le
  protocole, y compris ses refus.
- **Chaque anomalie a un test `xfail(strict=True)`** portant son identifiant.
- **Pas de dépendance native.** Le CI tourne sur un runner nu.
- **Les tests d'intégration utilisent les vrais modules.** Seules les
  frontières de processus sont simulées.

### Ce que la suite ne couvre pas

Hors périmètre automatisé, car cela suppose du matériel ou des poids de
modèles : qualité réelle de transcription, intelligibilité de la voix, latence
de bout en bout, micro physique, vrai serveur Ollama, et vrais serveurs MCP
tiers. Documenté dans [`TESTING.md`](TESTING.md), à passer avant chaque
version.
