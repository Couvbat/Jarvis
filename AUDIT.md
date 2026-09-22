# Jarvis — Rapport d'état du code

Audit du dépôt à la date du 11 septembre 2026 (commit `bc9a176`), mesuré
contre l'objectif du projet : un assistant vocal 100 % local avec tool
calling, CRUD système de fichiers et connexion MCP à des services externes.
Périmètre : les 9 modules Python, la configuration, les scripts d'installation
et la documentation. Chaque anomalie porte un identifiant `BUG-xx` repris
verbatim dans la suite de tests (`tests/`), sous forme de test `xfail(strict)`
qui passera au vert le jour où le correctif est appliqué.

> **Statut au 22 septembre 2026 — Phases 0 à 4 du plan livrées, plus les
> fournisseurs LLM ordonnés (3.5), le mot d'activation (5.1), la recherche
> documentaire (5.2) et les rappels (5.3).**
> **Les 34 anomalies de ce rapport sont corrigées et vérifiées par tests.**
> Il ne reste aucun marqueur `xfail`.
> Les trois anomalies bloquantes de la §2 et les quatre failles de la §3 sont
> closes. `action_executor.py` et `whitelist_manager.py` n'existent plus.
> Jarvis est client MCP, le pipeline est asynchrone et diffuse la parole
> pendant la génération, les conversations survivent aux redémarrages.
> Suite de tests : 953 passants, 0 `xfail`, 96 % de couverture.

**Résumé** : le squelette est complet et cohérent — les 7 modules existent,
s'assemblent proprement et la séparation des responsabilités est correcte.
Mais trois défauts bloquants font que le produit ne peut pas fonctionner
comme décrit : le VAD ne détecte jamais le silence, l'appel d'outils casse
avec un client Ollama récent, et l'installateur Piper produit une arborescence
que le module TTS n'accepte pas. Aucun test n'existait avant cet audit.

---

## 1. Ce qui est implémenté

| Module | État | Détail |
|---|---|---|
| `config.py` | ✅ Complet | Pydantic Settings, `.env`, propriétés dérivées |
| `whitelist_manager.py` | ✅ Complet | 3 catégories, persistance JSON, tolérant aux fichiers corrompus |
| `action_executor.py` | 🟡 Partiel | 3 outils sur 3 déclarés, sandbox de chemins solide |
| `llm_module.py` | 🟡 Partiel | Ollama + schéma d'outils, historique borné |
| `stt_module.py` | 🟡 Partiel | faster-whisper, changement de langue à chaud |
| `tts_module.py` | 🟡 Partiel | Piper via subprocess, découverte binaire/modèle |
| `audio_handler.py` | 🔴 Cassé | Enregistrement OK, **VAD inopérant** |
| `tui.py` | 🟡 Partiel | Layout Rich complet, confirmations interactives |
| `main.py` | 🟡 Partiel | Boucle vocale, mode texte, mode TUI |
| `setup_piper.py` | 🟡 Partiel | Linux x86_64/arm64, 1 voix anglaise |

### Fonctionnalités annoncées dans le README

| Annonce | Réalité |
|---|---|
| STT Whisper local | ✅ Implémenté |
| LLM Ollama local | ✅ Implémenté |
| TTS Piper | ✅ Implémenté (⚠️ voir BUG-15/16) |
| Détection d'activité vocale | ❌ **Code présent, jamais fonctionnel** (BUG-18) |
| Créer / lire / supprimer fichiers | ✅ Implémenté |
| **Éditer** des fichiers | ❌ Annoncé dans le prompt système, absent (BUG-05) |
| Supprimer un répertoire | ❌ Message d'erreur y renvoie, opération absente (BUG-03) |
| Récupération de pages web | ✅ Implémenté |
| **Recherche web** (« What's the weather? ») | ❌ Jamais implémenté, seulement `fetch_web_page` |
| Lancement d'applications | 🟡 Fonctionne sans arguments inline (BUG-02) |
| Confirmation de **toutes** les opérations fichier | ❌ Seulement `create_file` / `delete_file` (BUG-01) |
| Whitelist persistante | ✅ Implémenté |
| Mémoire conversationnelle | 🟡 En RAM uniquement, non persistée |
| Bascule FR/EN à la voix | 🟡 Mode vocal seulement (BUG-23) |
| `WHISPER_LANGUAGE=auto` | ❌ Documenté, non supporté par le code (BUG-12) |
| Mode texte « sans I/O audio » | 🟡 Charge quand même Whisper et Piper (BUG-24) |

### Infrastructure absente avant cet audit

- ❌ Aucun test (0 fichier)
- ❌ Aucune CI
- ❌ Aucun linter / formateur configuré
- ❌ Pas de `pyproject.toml`, pas de packaging, modules à plat
- ❌ Pas de `CHANGELOG`, pas de `LICENSE` (le README annonce MIT)
- ❌ Pas de versions épinglées (`ollama>=0.1.0` couvre des API incompatibles)

---

## 2. Anomalies bloquantes

### BUG-18 — Le VAD ne détecte jamais le silence
`audio_handler.py:62`

`webrtcvad` n'accepte que des trames de 10, 20 ou 30 ms. `CHUNK_SIZE=1024`
à 16 kHz fait 64 ms, donc **chaque** appel à `is_speech()` lève une exception,
avalée en `logger.debug`. Le compteur de silence n'avance jamais et chaque
énoncé enregistre les 30 secondes complètes de `max_duration` avant que
Whisper ne démarre.

*Effet utilisateur* : ~30 s de latence sur chaque tour de parole.
*Correctif* : `CHUNK_SIZE=320` (20 ms). La logique de silence est correcte —
les tests le prouvent en la faisant tourner avec une taille de trame légale.

### BUG-08 — L'appel d'outils casse avec un client Ollama récent
`llm_module.py:166`

```python
self.history.add_message("assistant", content if content else json.dumps(tool_calls))
```

Depuis ollama ≥ 0.4, `tool_calls` contient des modèles Pydantic, pas des dicts.
`json.dumps` lève `TypeError`. Comme la réponse à un appel d'outil a
typiquement un `content` vide, ce chemin est le cas **nominal**. L'exception
est attrapée par le `except Exception` de `chat()`, qui renvoie « I'm sorry,
I encountered an error » — aucun outil n'est jamais exécuté.

*Effet utilisateur* : toutes les actions système échouent silencieusement.
*Correctif* : sérialiser via `model_dump()` / stocker la structure telle quelle.

### BUG-15 — L'installateur Piper casse le module TTS
`tts_module.py:28` vs `setup_piper.py:63`

`setup_piper.py` extrait le binaire dans `piper/piper/piper`. La liste de
sondage de `_find_piper_binary()` contient `./piper/piper`, qui est alors un
**répertoire**. `subprocess.run()` lève `PermissionError`, non attrapé
(seuls `FileNotFoundError` et `TimeoutExpired` le sont) : construire
`TTSModule()` plante sur exactement l'arborescence que produit l'installateur
du projet.

*Effet utilisateur* : crash au démarrage après une installation « propre ».
*Correctif* : sonder `./piper/piper/piper` et attraper `OSError`.

---

## 3. Anomalies de sécurité

### BUG-01 — Les lectures ne sont pas confirmées
`action_executor.py:95`

Le README affirme « All file operations, web requests, and app launches require
user approval ». En réalité seuls `create_file` et `delete_file` demandent
confirmation. `read_file`, `list_directory` et `create_directory` s'exécutent
sans rien demander.

Avec `ALLOWED_DIRECTORIES=/home,/tmp` (la valeur par défaut), une injection de
prompt — dans une page web récupérée, par exemple — peut faire lire
`~/.ssh/id_rsa`, `~/.bash_history` ou `~/.config/**/credentials` et faire
énoncer le contenu à voix haute.

### BUG-04 — Pas de garde SSRF sur `fetch_web_page`
`action_executor.py:201`

Aucune validation de schéma ni d'hôte. `http://127.0.0.1:11434`,
`http://169.254.169.254/latest/meta-data/` ou l'interface d'admin d'un service
auto-hébergé sur le même réseau sont atteignables. La whitelist est par
domaine : approuver `127.0.0.1` une fois ouvre tous les ports locaux.
Aucune limite de taille n'est appliquée avant le téléchargement complet.

### BUG-32 — Extraction d'archive sans filtre
`setup_piper.py:60`

`tar.extractall(piper_dir)` sans `filter="data"`. Une archive forgée peut
écrire hors du répertoire cible (classe CVE-2007-4559). Combiné à BUG-31
(aucune vérification de somme de contrôle sur un téléchargement HTTPS suivi
d'un `chmod +x`), la chaîne d'installation n'a aucune garantie d'intégrité.

### BUG-06 — Whitelist de commandes comparée en texte brut
`action_executor.py:250`

`base_command not in self.command_whitelist` compare des chaînes. `/bin/ls`
est refusé alors que `ls` passe ; à l'inverse, un `ls` malveillant plus haut
dans le `PATH` est accepté sans distinction. La comparaison devrait porter sur
le nom de base résolu.

---

## 4. Anomalies fonctionnelles

| ID | Module | Description | Impact |
|---|---|---|---|
| BUG-02 | `action_executor` | `launch_application("ls -la")` : le whitelist découpe sur l'espace, l'exécution non → binaire littéral `"ls -la"` | Moyen |
| BUG-03 | `action_executor` | `delete_directory` référencé dans un message d'erreur, jamais implémenté | Faible |
| BUG-05 | `action_executor` | Aucune opération d'édition/ajout, contrairement au prompt système | Moyen |
| BUG-07 | `llm_module` | `messages[-0:]` renvoie toute la liste → `MAX_CONVERSATION_HISTORY=0` désactive la troncature | Faible |
| BUG-09 | `llm_module` | Les appels d'outils sont stockés en prose, pas au format protocole → le modèle ne peut pas corréler résultat et appel | Élevé |
| BUG-10 | `llm_module` | Aucun contrôle de disponibilité d'Ollama : une panne se manifeste par une erreur parlée à chaque tour | Moyen |
| BUG-11 | `llm_module` | `reset_conversation()` duplique le prompt système à chaque appel | Faible |
| BUG-12 | `stt_module` | `WHISPER_LANGUAGE=auto` documenté mais transmis littéralement à faster-whisper | Moyen |
| BUG-13 | `stt_module` | La langue détectée est journalisée puis jetée | Faible |
| BUG-14 | `stt_module` | Les segments Whisper commencent déjà par une espace ; `" ".join` la double | Faible |
| BUG-16 | `tts_module` | `synthesize()` jette la fréquence d'échantillonnage lue par `sf.read` | Moyen |
| BUG-17 | `tts_module` | Les fichiers temporaires fuient quand Piper échoue | Faible |
| BUG-19 | `audio_handler` | Le VAD reçoit l'audio multicanal entrelacé tel quel | Faible |
| BUG-20 | `audio_handler` | Aucun réglage de périphérique d'entrée | Moyen |
| BUG-21 | `audio_handler` | L'audio enregistré est 2-D `(frames, 1)` ; Whisper attend une onde mono 1-D | Élevé |
| BUG-22 | `main` | Les mots de sortie sont cherchés en sous-chaîne : « stop the music » quitte Jarvis | Élevé |
| BUG-23 | `main` | Le mode texte ne gère pas la bascule de langue | Faible |
| BUG-24 | `main` | `--text` charge quand même Whisper et Piper | Moyen |
| BUG-25 | `main` | Les options CLI inconnues sont ignorées en silence (`--txt` démarre le mode vocal) | Faible |
| BUG-26 | `main` | Le tour de suivi après outils est injecté comme faux message **utilisateur**, en anglais, et ses éventuels appels d'outils sont perdus | Élevé |
| BUG-27 | `main` | Lecture audio figée à 22050 Hz ; une voix Piper `*-low` est à 16 kHz | Moyen |
| BUG-28 | `tui` | L'historique croît sans borne en mémoire | Faible |
| BUG-29 | `tui` | `height=20` + 2 lignes par message → les **2 messages les plus récents** sont coupés | Élevé |
| BUG-30 | `tui` | Hauteurs de panneaux figées, débordent sous ~40 lignes | Faible |
| BUG-31 | `setup_piper` | Aucune vérification d'intégrité des téléchargements | Moyen |
| BUG-33 | `setup_piper` | Seule une voix anglaise est installée, alors que `WHISPER_LANGUAGE` vaut `fr` par défaut | Moyen |

### Incohérences de configuration

- `config.py` : `whisper_language="fr"` mais `piper_model="en_US-lessac-medium"`.
  Une installation par défaut écoute en français et répond avec une voix anglaise.
- `.env.example` : `COMMAND_WHITELIST` omet `code,firefox,nautilus` que
  `config.py` inclut dans ses valeurs par défaut et que `action_executor.py`
  traite comme applications graphiques.
- `README.md` §Installation : chemin codé en dur `/home/jules/Dev/other/Jarvis`.
- `README.md` §Project Structure : omet `tui.py` et `whitelist_manager.py`.

### Qualité de code

`ruff check .` avec le jeu de règles par défaut relève **429 problèmes** :
336 lignes vides contenant des espaces (W293), 27 annotations de typage
obsolètes (`List`/`Set` au lieu de `list`/`set`), 15 imports inutilisés,
12 blocs d'imports non triés. Aucun n'est fonctionnel, mais l'ensemble
empêche d'activer un linter en CI sans nettoyage préalable.

---

## 5. Ce que couvre la nouvelle suite de tests

306 tests passants, 38 `xfail(strict)` documentant les anomalies ci-dessus,
96 % de couverture de lignes, exécution en ~1,2 s.

```
action_executor.py       98%
audio_handler.py        100%
config.py               100%
llm_module.py           100%
main.py                  90%
stt_module.py            95%
tts_module.py            98%
tui.py                   96%
whitelist_manager.py    100%
-------------------------------
TOTAL                    96%
```

Les dépendances natives et réseau (`sounddevice`, `webrtcvad`,
`faster-whisper`, `ollama`) sont remplacées par des doublures fidèles dans
`tests/_stubs.py` : elles **rejettent ce que rejettent les vraies**
bibliothèques, ce qui est précisément ce qui permet à la suite de mettre
BUG-18 et BUG-08 en évidence. La suite tourne donc sans PortAudio, sans
compilateur C, sans téléchargement de modèle et sans serveur Ollama.

---

## 6. Écart avec l'objectif final

L'objectif du projet est un assistant vocal 100 % local avec tool calling,
CRUD système de fichiers et connexion MCP à des services externes. Mesuré
contre cette cible — et non contre le README — voici ce qui manque.

### Acquis

- Chaîne STT → LLM → TTS entièrement locale : les briques sont là et bien
  choisies (faster-whisper, Ollama, Piper).
- Sandbox de chemins solide : résolution des symlinks, traversée `..`
  bloquée, préfixes voisins correctement rejetés. C'est la partie la mieux
  écrite du dépôt.
- Confirmation avec approbation persistante : le modèle est le bon, il est
  seulement appliqué trop partiellement.

### Manquant

| Brique de l'objectif | État |
|---|---|
| Tool calling | 🔴 **Cassé** (BUG-08), et le format d'historique n'est pas celui du protocole (BUG-09) |
| **C**reate système de fichiers | ✅ |
| **R**ead système de fichiers | 🟡 Pas de pagination ni de lecture par plage — un gros fichier ne tient pas dans le contexte |
| **U**pdate système de fichiers | ❌ **Absent** — aucune écriture partielle, aucun ajout, aucun remplacement |
| **D**elete système de fichiers | 🟡 Fichiers seulement, pas de répertoires |
| Déplacement, copie, recherche | ❌ Absent |
| **Connexion MCP** | ❌ **Absent** — aucune trace dans le dépôt |
| Registre d'outils extensible | ❌ Liste codée en dur + dispatch en `if/elif` |
| Pipeline asynchrone | ❌ Entièrement synchrone — le SDK MCP est async |
| Génération et synthèse en flux | ❌ Pipeline strictement séquentiel (~3-8 s avant le premier son) |
| Interruption pendant la réponse | ❌ Absent |
| Persistance des conversations | ❌ Mémoire vive uniquement |

### Les trois obstacles structurels

1. **`action_executor.py` est un cul-de-sac architectural.** Liste d'outils
   codée en dur dans `llm_module.py`, dispatch en `if/elif`, politique de
   confirmation mêlée à l'exécution. Chaque nouvelle capacité passe par ce
   fichier. MCP, qui apporte des dizaines d'outils découverts à l'exécution,
   n'a aucun point d'accroche.

2. **Le code est synchrone, le SDK MCP est asynchrone.** `ClientSession`,
   `stdio_client` et `ClientSessionGroup` sont des gestionnaires de contexte
   async. Le streaming et l'interruption exigeront de toute façon
   l'asynchrone : le passage doit se faire avant de brancher MCP, pas pendant.

3. **Le modèle de sécurité ne tient pas à l'échelle de MCP.** Aujourd'hui les
   trois outils sont écrits dans le dépôt et audités. Un serveur MCP est un
   tiers dont les *descriptions d'outils* entrent dans le prompt et dont les
   résultats reviennent dans la conversation. Combiné à l'accès fichiers et à
   la récupération web, Jarvis réunira les trois conditions qui rendent
   l'injection de prompt exploitable — avec un modèle local, qui y résiste
   moins bien qu'un modèle frontière.

**À noter** : ces failles ne sont pas exploitables aujourd'hui, puisque
BUG-08 fait qu'aucun outil ne s'exécute jamais. Corriger BUG-08 les rend
atteignables. C'est la raison pour laquelle le plan lie les Phases 0 et 1.

Voir [`ARCHITECTURE.md`](ARCHITECTURE.md) pour l'architecture cible et
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) pour le chemin.
