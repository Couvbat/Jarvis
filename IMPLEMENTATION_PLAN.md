# Jarvis — Plan d'implémentation

**Cible** : assistant vocal 100 % local, tool calling, CRUD système de
fichiers, connexion MCP à des services externes.
Architecture visée : [`ARCHITECTURE.md`](ARCHITECTURE.md).
État actuel : [`AUDIT.md`](AUDIT.md).

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
- `mcp>=2.2,<3` dans `requirements.txt`.

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

## Phase 4 — Qualité et packaging (≈ 2 jours)

- **Nettoyage lint** en un commit isolé : `ruff check --fix .` puis
  `ruff format .` (429 problèmes, dont 336 lignes vides avec espaces). Ensuite
  élargir `select` dans `ruff.toml` à `["E", "F", "W", "I", "UP", "B"]` et
  retirer l'`ignore` sur B904.
- **Packaging** : `pyproject.toml`, modules sous `src/jarvis/`, point d'entrée
  `jarvis = "jarvis.__main__:main"`. Versions majeures épinglées.
- **`LICENSE`** : le README annonce MIT, le fichier n'existe pas.
- **`TESTING.md`** : la campagne manuelle que la suite automatisée ne peut pas
  couvrir (qualité de transcription réelle, intelligibilité de la voix,
  latence mesurée, micro physique, vrai serveur Ollama).

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
| #16 Meilleure récupération sur erreur | Réparti sur 0→3 | Largement traité en chemin |
| #1 Mot-clé d'activation | **Après Phase 3** | Exige un VAD fonctionnel (Phase 0.1) et une boucle audio continue (barge-in, Phase 3). Placé en premier dans `FEATURES_IDEA.md`, mais l'implémenter avant reviendrait à bâtir sur du sable. **Porcupine exige une clé d'API Picovoice** — incompatible avec l'objectif local ; évaluer `openWakeWord`. |
| #2 RAG documentaire | Après MCP | Partage les embeddings avec la sélection d'outils (Phase 2.4), ce qui en réduit le coût. Alternative : un serveur MCP dédié, au prix d'un peu de latence. |
| #11 Vision (LLaVA) | Plus tard | Ollama gère les modèles multimodaux ; le coût est surtout en RAM |
| #9 Multi-utilisateur | Plus tard | Suppose un modèle de permissions par utilisateur — un projet en soi |
| #13 API / WebSocket | Plus tard | Devient « Jarvis exposé *comme* serveur MCP », plus intéressant que REST, mais orthogonal au rôle de client |

---

## Charge totale

| Phase | Charge |
|---|---|
| 0 — Fonctionnel | 1 j |
| 1 — Registre, CRUD, politique | 4 j |
| 2 — Async et MCP | 5 j |
| 3 — Latence et usage quotidien | 4 j |
| 4 — Qualité et packaging | 2 j |
| **Jusqu'à l'objectif affiché** | **~16 j** |

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
tiers. À documenter dans `TESTING.md` en Phase 4.
