# Jarvis — Plan d'implémentation

Suite de [`AUDIT.md`](AUDIT.md). Objectif : passer d'un squelette complet mais
non fonctionnel à un assistant utilisable au quotidien, puis dérouler les
fonctionnalités de [`FEATURES_IDEA.md`](FEATURES_IDEA.md).

**Règle de sortie de chaque tâche** : le ou les tests `xfail(strict)`
correspondants passent au vert. `pytest` échoue si un `xfail` strict réussit
sans qu'on ait retiré le marqueur — le correctif et son test avancent donc
ensemble par construction.

Charges indiquées pour un développeur familier du dépôt.

---

## Phase 0 — Rendre le produit fonctionnel (≈ 1 journée)

Sans ces cinq correctifs, Jarvis ne fait pas ce que le README décrit.

### 0.1 — VAD : taille de trame légale · BUG-18 · 30 min
`config.py`, `.env.example`, `audio_handler.py`

- `chunk_size: int = 320` (20 ms à 16 kHz), aligner `.env.example`.
- Valider la trame à la construction : `chunk_size * 1000 / sample_rate`
  doit valoir 10, 20 ou 30, sinon lever une erreur explicite au démarrage
  plutôt que d'avaler l'exception par trame.
- Remonter l'échec VAD en `logger.warning` une seule fois, pas en `debug`
  à chaque trame.

*Tests* : `test_silence_detection_works_with_the_default_chunk_size`.
La logique de silence elle-même est déjà couverte par `TestVoiceActivityDetection`,
qui tourne aujourd'hui avec `chunk_size=320` — elle passe déjà.

### 0.2 — Audio mono 1-D vers Whisper · BUG-21 · 20 min
`audio_handler.py`

`np.concatenate(frames)` produit `(n, channels)`. Ajouter un aplatissement
mono en sortie de `record_until_silence` (`audio.mean(axis=1)` en `int16`,
ou `[:, 0]` si `channels == 1`), et alimenter le VAD avec le canal 0
uniquement (BUG-19).

*Tests* : `test_recording_returns_a_mono_waveform`,
`test_stereo_capture_still_feeds_mono_frames_to_the_vad`.

### 0.3 — Protocole d'appel d'outils Ollama · BUG-08, BUG-09, BUG-26 · 3 h
`llm_module.py`, `main.py`

C'est le correctif structurant de la phase.

1. `ConversationHistory.add_message` accepte des champs additionnels
   (`tool_calls`, `tool_name`) au lieu d'un simple couple rôle/contenu.
2. `chat()` stocke le tour assistant sous la forme attendue par Ollama :
   `{"role": "assistant", "content": ..., "tool_calls": [...]}`, en
   normalisant les modèles Pydantic via `model_dump()`. Plus de `json.dumps`.
3. `add_tool_result()` émet `{"role": "tool", "content": ..., "name": ...}`.
4. Remplacer le faux tour utilisateur « Please provide a natural response… »
   par une méthode dédiée `continue_after_tools()` qui rappelle le modèle sur
   l'historique existant, sans injecter de message, et qui **exécute** les
   appels d'outils éventuels du second tour (boucle bornée, 3 itérations max).
5. Épingler `ollama>=0.4,<1.0` dans `requirements.txt`.

*Tests* : `test_tool_call_without_content_survives_history_recording`,
`test_tool_calls_are_replayed_in_the_protocol_shape`,
`test_follow_up_after_tools_is_not_a_user_turn`,
`test_file_creation_works_with_a_modern_ollama_client`.

### 0.4 — Découverte du binaire Piper · BUG-15, BUG-16, BUG-27 · 1 h
`tts_module.py`, `main.py`

- Ajouter `./piper/piper/piper` à la liste de sondage et attraper `OSError`
  (qui couvre `PermissionError`, `IsADirectoryError`, `FileNotFoundError`).
- `synthesize()` renvoie `(audio, sample_rate)`. Mettre à jour les 5 appels
  de `main.py` pour passer ce taux à `play_audio` au lieu de `22050`.
- Envelopper les fichiers temporaires dans `try/finally` (BUG-17).

*Tests* : `test_local_piper_directory_does_not_break_construction`,
`test_synthesize_reports_its_sample_rate`,
`test_temp_files_are_cleaned_up_when_piper_fails`,
`test_playback_uses_the_synthesiser_sample_rate`.

### 0.5 — Détection de la commande de sortie · BUG-22 · 30 min
`main.py`

Comparer sur le texte normalisé complet (ponctuation retirée, minuscules),
pas en sous-chaîne. Accepter une courte liste de formulations exactes
(`exit`, `quit`, `goodbye`, `au revoir`, `arrête-toi`) plutôt que
`any(cmd in lower_text …)`.

*Tests* : `test_a_sentence_containing_stop_is_not_an_exit_command`.

---

## Phase 1 — Sécurité (≈ 1 journée)

### 1.1 — Confirmer toutes les opérations fichier · BUG-01 · 2 h
`action_executor.py`

Faire passer `read_file`, `list_directory` et `create_directory` par
`_request_confirmation`. Conserver des catégories de whitelist distinctes par
opération pour qu'approuver une lecture n'approuve pas une suppression.
Envisager un mode `read_only_auto_approve` configurable pour les utilisateurs
qui acceptent le compromis — mais le défaut doit correspondre au README.

*Tests* : les trois `test_*_should_require_confirmation`.

### 1.2 — Garde SSRF et limite de taille · BUG-04 · 3 h
`action_executor.py`

- Rejeter tout schéma autre que `http`/`https`.
- Résoudre l'hôte et refuser les plages loopback, link-local (169.254.0.0/16),
  privées (RFC 1918) et `::1`, sauf autorisation explicite en configuration
  (`ALLOW_PRIVATE_NETWORK_FETCH=false` par défaut).
- `stream=True` + coupure au-delà de `MAX_FETCH_BYTES` (défaut 2 Mo) avant
  d'appeler BeautifulSoup.
- Refuser les redirections qui changent d'hôte vers une plage interdite.

*Tests* : `test_non_http_scheme_should_be_rejected`,
`test_loopback_url_should_be_rejected`,
`test_oversized_response_should_be_refused_before_parsing`.

### 1.3 — Whitelist de commandes par nom de base résolu · BUG-06, BUG-02 · 2 h
`action_executor.py`

- Découper la commande avec `shlex.split()` (l'import est déjà là, inutilisé).
- Comparer `Path(shutil.which(cmd)).name` à la whitelist.
- Refuser les chemins relatifs (`./rm`).

*Tests* : `test_command_string_containing_arguments_should_run`,
`test_absolute_path_to_a_listed_command_should_be_normalised`.

### 1.4 — Chaîne d'installation vérifiable · BUG-31, BUG-32 · 2 h
`setup_piper.py`

- `tar.extractall(piper_dir, filter="data")`.
- Sommes SHA-256 épinglées pour le binaire et les modèles, vérifiées avant
  extraction et avant `chmod +x`.
- Échec bruyant si la somme ne correspond pas ; ne pas laisser d'archive
  partielle derrière soi.

*Tests* : `test_downloads_are_verified`,
`test_extraction_refuses_paths_outside_the_target`.

### 1.5 — Restreindre les valeurs par défaut · 1 h
`config.py`, `.env.example`, `README.md`

`ALLOWED_DIRECTORIES=/home,/tmp` donne accès à `~/.ssh`, `~/.aws`,
`~/.config`. Proposer `~/Documents,~/Downloads,/tmp` par défaut, et
documenter explicitement le risque d'élargir. Ajouter une liste
`DENIED_PATTERNS` (`.ssh`, `.aws`, `.gnupg`, `*_history`, `.env`) évaluée
après la validation de sandbox.

---

## Phase 2 — Combler les écarts avec la documentation (≈ 1 journée)

| Tâche | Bugs | Charge |
|---|---|---|
| Opérations `delete_directory` et `append_file`/`edit_file` (+ schéma d'outil) | BUG-03, BUG-05 | 3 h |
| `WHISPER_LANGUAGE=auto` → `language=None` ; exposer la langue détectée | BUG-12, BUG-13 | 1 h |
| Jointure des segments Whisper sans double espace | BUG-14 | 15 min |
| Bascule de langue en mode texte (extraire la logique de `run_interactive`) | BUG-23 | 1 h |
| `--text` : ne charger ni Whisper ni Piper | BUG-24 | 1 h |
| `argparse` à la place du parsing manuel ; combiner `--tui` et `--text` | BUG-25 | 1 h |
| `reset_conversation` sans doublon de prompt système ; garde `max_history<=0` | BUG-07, BUG-11 | 30 min |
| Contrôle de disponibilité d'Ollama au démarrage, message clair | BUG-10 | 1 h |
| Sélection du périphérique d'entrée (`AUDIO_INPUT_DEVICE`) | BUG-20 | 1 h |
| Voix française dans l'installateur ; alignement `WHISPER_LANGUAGE`/`PIPER_MODEL` | BUG-33 | 2 h |
| Corrections README (chemin en dur, structure du projet, recherche web) | — | 1 h |

---

## Phase 3 — TUI et qualité (≈ 1 journée)

- **BUG-29** — Panneau de conversation : passer à une hauteur relative
  (`ratio`) ou rendre un nombre de messages calculé à partir de la hauteur
  disponible, pour que la réponse la plus récente soit toujours visible.
  C'est le bug le plus visible à l'usage du mode TUI.
- **BUG-30** — Retirer les `height=` codés en dur, laisser `Layout` répartir.
- **BUG-28** — Borner `chat_history` et `actions_log` (`collections.deque`
  avec `maxlen`).
- **Nettoyage lint** — `ruff check --fix .` puis `ruff format .` (429
  problèmes, dont 336 W293). Ensuite élargir `select` dans `ruff.toml` à
  `["E", "F", "W", "I", "UP", "B"]` et retirer l'`ignore` sur B904.
  À faire en un commit isolé pour ne pas mélanger style et logique.
- **Packaging** — `pyproject.toml`, passage des modules sous `src/jarvis/`,
  point d'entrée `jarvis = "jarvis.main:main"`. Épingler les versions
  majeures dans `requirements.txt`.
- **`LICENSE`** — Le README annonce MIT, le fichier n'existe pas.

---

## Phase 4 — Fonctionnalités de `FEATURES_IDEA.md`

L'ordre proposé dans ce document est bon ; deux ajustements au vu de l'audit.

**À avancer** — *Persistance des conversations* (feature #3, estimée 2-3 h).
Elle devient quasi gratuite une fois la Phase 0.3 faite : l'historique aura
déjà la bonne structure, il ne reste qu'à le sérialiser. C'est aussi le
prérequis de tout le reste (RAG, analytics, multi-utilisateur).

**À avancer** — *Meilleure récupération sur erreur* (feature #16). La moitié
du travail est déjà dans les Phases 0 et 2 (contrôle Ollama, échec VAD
explicite, gestion Piper). Terminer par des relances avec back-off sur les
requêtes web et un mode dégradé sans TTS.

**À retarder** — *Détection de mot-clé d'activation* (feature #1, placée en
premier dans `FEATURES_IDEA.md`). Elle repose sur une boucle audio continue,
donc sur un VAD qui fonctionne : sans la Phase 0.1, l'implémenter revient à
bâtir sur du sable. Porcupine ajoute par ailleurs une dépendance à une clé
d'API Picovoice, ce qui frotte avec le positionnement « local-first » —
`openWakeWord` est une alternative entièrement locale à évaluer.

**Prérequis transverse** — Le *système de plugins* (feature #4) devrait
précéder l'ajout de nouvelles catégories d'outils (git/docker, email/agenda,
supervision système). Sinon `action_executor.py`, déjà à 400 lignes, devient
le point de passage obligé de toutes les fonctionnalités à venir.

---

## Suite de tests

### Exécution

```bash
pip install -r requirements-test.txt   # pas de PortAudio, pas de modèle, pas d'Ollama
pytest                                 # ~1,2 s
pytest --cov --cov-report=term-missing
ruff check .
```

### Organisation

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

### Principes retenus

- **Les doublures refusent ce que refusent les vraies bibliothèques.** La
  doublure `webrtcvad` rejette les trames de 64 ms, la doublure `ollama`
  renvoie des objets non sérialisables en JSON. Une doublure permissive aurait
  masqué BUG-18 et BUG-08, qui sont les deux anomalies les plus graves du dépôt.
- **Chaque anomalie a un test `xfail(strict=True)`** portant son identifiant et
  expliquant l'effet concret. La suite est verte aujourd'hui *et* signale toute
  régression comme tout correctif : un `xfail` strict qui réussit fait échouer
  `pytest`, donc on ne peut pas corriger un bug sans retirer son marqueur.
- **Pas de dépendance native.** Ni PortAudio, ni compilateur C, ni
  téléchargement de modèle, ni serveur Ollama : la CI tourne sur un runner nu.
- **Les tests d'intégration utilisent les vrais modules.** Seules les
  frontières de processus sont simulées.

### CI

`.github/workflows/tests.yml` exécute la suite sur Python 3.10, 3.11 et 3.12,
plus un job `ruff`. La couverture est publiée en artefact.

### Ce que la suite ne couvre pas

Volontairement hors périmètre, car cela suppose du matériel ou des poids de
modèles : la qualité réelle de transcription Whisper, l'intelligibilité de la
voix Piper, la latence de bout en bout, le comportement avec un microphone
physique, et les réponses d'un vrai serveur Ollama. Ces points relèvent d'une
campagne manuelle à documenter dans un `TESTING.md` lors de la Phase 3.
