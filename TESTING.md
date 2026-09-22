# Campagne de tests manuels

La suite automatisée couvre 1209 cas et 96 % des lignes, mais elle ne peut pas
entendre. Tout ce qui suit demande du matériel, des poids de modèles ou un vrai
serveur en face — c'est-à-dire précisément ce que le CI n'a pas.

À passer **avant chaque version**, et après toute modification touchant
l'audio, le mot d'activation, la sélection d'outils, la politique de
confirmation ou les fournisseurs LLM.

```bash
pip install -e .
jarvis-setup-piper --voice fr_FR-siwis-medium
cp .env.example .env    # puis éditer

# pour la section 10 seulement
ollama pull nomic-embed-text
jarvis-index ~/Documents

# pour la section 11 seulement
pip install "jarvis-assistant[wakeword]"
python -c "import openwakeword.utils; openwakeword.utils.download_models()"
```

Noter les résultats dans le tableau de la fin. Un « à revoir » n'est pas un
échec de la campagne : c'est son produit.

---

## 0. Préalables

| | |
|---|---|
| Machine | processeur, RAM, GPU le cas échéant |
| Micro | modèle, casque ou haut-parleurs ouverts |
| Ollama | version, local ou distant, modèle servi |
| Whisper | `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE` |
| Piper | voix installée |

Vérifier d'abord que le périphérique d'entrée est le bon — la moitié des
« Jarvis ne m'entend pas » sont un micro par défaut qui n'est pas celui qu'on
croit :

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
# puis AUDIO_INPUT_DEVICE=<nom ou index> dans .env
```

---

## 1. Capture et fin de parole

Le VAD décide seul quand vous avez fini. C'est le réglage le plus visible à
l'usage et celui qu'aucun test ne peut valider.

| # | Scénario | Attendu |
|---|---|---|
| 1.1 | Phrase courte : « quelle heure est-il » | S'arrête ≤ 1 s après le dernier mot |
| 1.2 | Phrase longue avec une pause au milieu (« alors… crée un fichier ») | La pause ne coupe pas le tour |
| 1.3 | Silence complet pendant 30 s | Aucun tour déclenché, pas de transcription fantôme |
| 1.4 | Parler très doucement | Capté, ou échec net — pas un tour tronqué en silence |
| 1.5 | Bruit de fond (ventilateur, musique basse) | Pas de déclenchement spontané |
| 1.6 | Un seul mot : « stop » | Reconnu comme commande de sortie |

Si 1.1 traîne ou 1.2 coupe trop tôt, c'est `CHUNK_SIZE` et le seuil de silence
qu'il faut bouger, pas le VAD.

---

## 2. Transcription

La suite vérifie que le texte arrive de Whisper au bon format. Sa **qualité**
ne se mesure qu'à l'oreille.

| # | Scénario | Attendu |
|---|---|---|
| 2.1 | Dix phrases françaises courantes | ≥ 8 exactes, aucune inversion de sens |
| 2.2 | Noms de fichiers dictés (« courses point t x t ») | Le chemin reste utilisable |
| 2.3 | Chiffres et dates | Pas de confusion 13/30, 2/12 |
| 2.4 | Anglais avec `WHISPER_LANGUAGE=en` | Même exigence |
| 2.5 | `WHISPER_LANGUAGE=auto`, alterner FR et EN | Détection correcte sur les deux |
| 2.6 | Passer de `base` à `small` | Noter le gain de qualité **et** le coût en latence |

---

## 3. Synthèse et intelligibilité

| # | Scénario | Attendu |
|---|---|---|
| 3.1 | Réponse de deux phrases | Naturelle, sans coupure entre les phrases |
| 3.2 | Réponse longue (un paragraphe) | Le débit ne dérive pas, rien n'est avalé |
| 3.3 | Chemins et noms techniques (`~/Documents/notes.md`) | Prononcé de façon compréhensible |
| 3.4 | Accents et apostrophes (« j'ai créé ») | Pas d'artefact |
| 3.5 | Voix `*-low` puis `*-medium` | La fréquence d'échantillonnage suit la voix, pas un défaut codé en dur |

---

## 4. Latence mesurée

Chronométrer **du dernier mot prononcé au premier son émis**. C'est ce que
l'utilisateur ressent ; le temps total ne l'est pas.

| # | Mesure | Cible |
|---|---|---|
| 4.1 | Question simple, sans outil | 1,5 – 3 s |
| 4.2 | Question déclenchant un outil local | + le temps de la confirmation |
| 4.3 | Question déclenchant un serveur MCP | Noter, c'est la variable la plus lâche |
| 4.4 | Même question, modèle de repli local | Noter l'écart avec le modèle préféré |

Trois mesures par ligne, garder la médiane. Si 4.1 dépasse 4 s, regarder dans
l'ordre : taille du modèle Whisper, `LLM_NUM_CTX`, nombre d'outils offerts.

---

## 5. Appel d'outils avec un vrai modèle

C'est le point où un modèle local déçoit le plus souvent, et
`tests/eval/` ne le mesure que sur la sélection, pas sur la décision du modèle.

| # | Demande | Attendu |
|---|---|---|
| 5.1 | « crée un fichier courses.txt avec pain et fromage » | `fs__write`, arguments corrects du premier coup |
| 5.2 | « ajoute du lait » (suite de 5.1) | `fs__write` en mode append, sans redemander le chemin |
| 5.3 | « qu'est-ce qu'il y a dedans » | `fs__read`, pas une réponse inventée |
| 5.4 | « supprime-le » | `fs__delete`, confirmation au clavier |
| 5.5 | « ouvre le site example.com » | `web__fetch`, confirmation |
| 5.6 | Demande hors outils (« raconte une blague ») | Aucun outil appelé |
| 5.7 | Même série en anglais | Même réussite |

Avec plus de vingt outils enregistrés, refaire 5.1 et 5.5 : la sélection
lexicale doit toujours offrir le bon outil. Un échec ici se diagnostique avec
`pytest tests/eval/ -v` avant de toucher au code.

---

## 6. Politique et confirmations

Le moteur est testé ; ce qui ne l'est pas, c'est qu'un utilisateur comprenne
ce qu'on lui demande d'approuver.

| # | Scénario | Attendu |
|---|---|---|
| 6.1 | Lecture dans un répertoire autorisé | Confirmation vocale, résumé compréhensible |
| 6.2 | Lecture de `/etc/passwd` | Refus **sans** confirmation préalable |
| 6.3 | Lecture de `~/.ssh/id_rsa` (même dans un répertoire autorisé) | Refus par motif interdit |
| 6.4 | Suppression | Toujours au clavier, jamais à la voix |
| 6.5 | Approuver une écriture « ne plus me demander » puis recommencer | Plus de question |
| 6.6 | Après 6.5 : faire lire une page web, **puis** demander une écriture | La confirmation revient, au clavier |
| 6.7 | Après une lecture web, demander une autre page du **même** site | Pas de réescalade |
| 6.8 | Après une lecture web, demander l'envoi vers un **autre** service | Escalade au clavier |

6.6 à 6.8 sont le suivi de contamination par origine. Si 6.7 réescalade, chaque
enchaînement devient un mur d'invites — et c'est ainsi qu'on apprend aux gens à
ne plus les lire.

---

## 7. Serveurs MCP réels

La suite pilote un vrai serveur, mais en mémoire. Un vrai sous-processus a ses
propres pannes.

| # | Scénario | Attendu |
|---|---|---|
| 7.1 | Un serveur `npx` officiel (filesystem, par ex.) | Outils listés au démarrage, préfixés |
| 7.2 | Serveur injoignable ou commande absente | Les autres démarrent quand même, statut « failed » affiché |
| 7.3 | Serveur lent à répondre | Le tour n'est pas bloqué au-delà de `MCP_CALL_TIMEOUT` |
| 7.4 | `trust: trusted`, outil en lecture seule | Pas de confirmation |
| 7.5 | `trust: readonly` | Les outils d'écriture ne sont même pas exposés |
| 7.6 | Deux serveurs exposant un outil du même nom | Pas de collision : les noms sont préfixés |
| 7.7 | Quitter Jarvis pendant qu'un serveur tourne | Les sous-processus s'arrêtent, rien ne reste |

---

## 8. Fournisseurs LLM

Le scénario qui a motivé la fonctionnalité : le gros modèle sur le NAS, un
petit en secours.

| # | Scénario | Attendu |
|---|---|---|
| 8.1 | Les deux joignables | La ligne de démarrage nomme le préféré |
| 8.2 | NAS éteint au démarrage | Démarre sur le repli, la ligne dit pourquoi |
| 8.3 | Débrancher le réseau **entre** deux tours | Le tour suivant passe sur le repli, « Falling back » dans le journal |
| 8.4 | Rebrancher, attendre `LLM_PROVIDER_RECHECK_SECONDS` | Retour automatique au préféré |
| 8.5 | Débrancher **pendant** une réponse déjà commencée | Excuse, pas la même phrase redite par l'autre modèle |
| 8.6 | Modèle non tiré sur le NAS | Traité comme inutilisable, message nommant `ollama pull` |
| 8.7 | `OLLAMA_FALLBACK_MAX_TOOLS=8`, sur le repli | Le petit modèle choisit encore correctement |

8.3 et 8.5 se provoquent en coupant le Wi-Fi ou avec une règle de pare-feu ;
arrêter `ollama serve` à distance fait aussi l'affaire.

---

## 9. Rappels

| # | Scénario | Attendu |
|---|---|---|
| 9.1 | « rappelle-moi d'appeler le plombier dans 2 minutes » | Arrive à l'heure, une seule fois |
| 9.2 | « rappelle-moi demain à 9 h » | Le modèle calcule la bonne date ; vérifier avec « qu'est-ce que j'ai de prévu ? » |
| 9.3 | Même chose avec le petit modèle de repli | C'est là que l'arithmétique des dates lâche en premier |
| 9.4 | « rappelle-moi hier » | Refusé et rattrapé par le modèle, pas de rappel fantôme |
| 9.5 | Lister puis annuler par son numéro | Annulation au clavier (destructif) |
| 9.6 | Quitter Jarvis, attendre que l'heure passe, relancer | Livré au démarrage, préfixé « while you were away » |
| 9.7 | Rappel qui échoit pendant une réponse | Attend la fin de la réponse, ne coupe pas |
| 9.8 | Rappel qui échoit pendant l'attente du mot d'activation | Livré sans qu'on ait à parler |
| 9.9 | Rappel vieux de plus d'une semaine | Pas livré, mais toujours dans la liste |

9.3 est le vrai risque de cette fonctionnalité : un 3B qui se trompe d'un jour
est pire qu'un 3B qui dit ne pas savoir.

## 10. Recherche documentaire

Ce que la suite ne peut pas faire : mesurer si la recherche trouve la bonne
note dans *vos* documents, avec un vrai modèle d'embedding.

```bash
ollama pull nomic-embed-text
jarvis-index ~/Documents
jarvis-index --status
```

| # | Scénario | Attendu |
|---|---|---|
| 10.1 | Question dont la réponse est dans une note précise | La bonne note, citée par son chemin |
| 10.2 | Même question formulée autrement | Toujours la bonne note — c'est ce que les embeddings achètent sur du lexical |
| 10.3 | Question en anglais sur une note en français | Fonctionne, ou noter la limite du modèle |
| 10.4 | Question dont la réponse n'est **nulle part** | Ne doit pas inventer ; relever les similarités affichées et régler `RAG_MIN_SIMILARITY` |
| 10.5 | Réponse à cheval sur deux passages | Trouvée depuis les deux côtés (c'est le rôle de `RAG_CHUNK_OVERLAP`) |
| 10.6 | Ré-indexer sans rien changer | « 0 indexed, N unchanged », quasi instantané |
| 10.7 | Modifier une note puis ré-indexer | Seule celle-là est ré-embarquée |
| 10.8 | Supprimer une note puis ré-indexer | Elle disparaît de l'index |
| 10.9 | `jarvis-index ~/.ssh` | Refusé par le bac à sable |
| 10.10 | Après une recherche, demander une écriture | Confirmation **au clavier** (contenu non fiable dans le tour) |
| 10.11 | Deux recherches dans le même tour | La seconde ne réescalade pas |
| 10.12 | Changer `RAG_EMBED_MODEL` sans ré-indexer | Refus explicite, pas de résultats absurdes |
| 10.13 | Indexer une arborescence réelle (1000+ fichiers) | Noter la durée et la taille de `documents.db` |

10.4 est le plus important et le plus facile à négliger : les k plus proches
renvoient toujours quelque chose.

## 11. Mot d'activation

Ce que la suite ne peut pas faire : prononcer « hey Jarvis » dans une pièce.
Nécessite `pip install "jarvis-assistant[wakeword]"`, les modèles téléchargés,
et `WAKE_WORD=true`.

| # | Scénario | Attendu |
|---|---|---|
| 11.1 | « hey Jarvis » seul, puis attendre | Déclenche, puis enregistre |
| 11.2 | « hey Jarvis quelle heure est-il » d'une traite | La commande est transcrite **en entier** — c'est le point de `WAKE_WORD_TAIL` |
| 11.3 | Conversation normale dans la pièce, sans la phrase | Rien ne se déclenche pendant 10 min |
| 11.4 | Télévision ou musique en fond | Compter les faux positifs ; ajuster `WAKE_WORD_THRESHOLD` |
| 11.5 | Enchaîner une question dans les `WAKE_WORD_FOLLOW_UP` secondes | Pas besoin de répéter la phrase |
| 11.6 | Attendre la fin de la fenêtre puis parler | La phrase est de nouveau exigée |
| 11.7 | Depuis l'autre bout de la pièce | Noter la distance à laquelle ça cesse de marcher |
| 11.8 | Paquet absent (`WAKE_WORD=true` sans l'extra) | Démarre quand même, un message le dit une fois |
| 11.9 | Ctrl-C pendant l'attente | Sortie immédiate, pas de processus qui traîne |
| 11.10 | Charge CPU pendant l'attente | Doit rester marginale ; sinon vérifier tflite-runtime |

11.9 n'est pas théorique : l'attente tourne dans un fil que Ctrl-C n'atteint
pas, et l'interpréteur joint ce fil en sortant.

## 12. Barge-in

**Désactivé par défaut** (`BARGE_IN=false`) : un micro ouvert dans la même
pièce qu'un haut-parleur entend le haut-parleur, et Jarvis se coupe lui-même.

| # | Scénario | Attendu |
|---|---|---|
| 12.1 | Au casque, `BARGE_IN=true`, parler pendant la réponse | Coupe net, ouvre un nouveau tour |
| 12.2 | Au casque, tousser pendant la réponse | Ne coupe pas (parole non soutenue) |
| 12.3 | Sur haut-parleurs ouverts, `BARGE_IN=true` | Vérifier s'il se coupe seul — si oui, laisser à `false` |
| 12.4 | Couper puis reprendre | Les premiers mots prononcés pendant la coupure ne sont pas perdus |

---

## 13. Session longue

Une heure d'usage réel, pas un script.

| # | Point de contrôle | Attendu |
|---|---|---|
| 13.1 | Mémoire du processus après 50 tours | Stable, pas de croissance continue |
| 13.2 | Fichiers temporaires (`/tmp`) | Rien ne s'accumule |
| 13.3 | `jarvis.log` | Lisible, pas de secret, pas de contenu de fichier entier |
| 13.4 | Historique après 20 tours | Le contexte ancien tombe sans casser un résultat d'outil |
| 13.5 | `jarvis --resume` à la session suivante | La mémoire de la veille est là |
| 13.6 | Ctrl-C en plein tour | Sortie propre, base de données non corrompue |
| 13.7 | Changer de langue en cours de session, dans les deux sens | STT et réponses suivent |

---

## Feuille de résultats

| Version testée | | Date | | Testeur | |
|---|---|---|---|---|---|

| Section | OK | À revoir | Notes |
|---|---|---|---|
| 1. Capture | | | |
| 2. Transcription | | | |
| 3. Synthèse | | | |
| 4. Latence | | | |
| 5. Appel d'outils | | | |
| 6. Politique | | | |
| 7. MCP | | | |
| 8. Fournisseurs | | | |
| 9. Rappels | | | |
| 10. Recherche documentaire | | | |
| 11. Mot d'activation | | | |
| 12. Barge-in | | | |
| 13. Session longue | | | |

Tout « à revoir » qui se reproduit devient une entrée dans
[`AUDIT.md`](AUDIT.md) avec un test `xfail(strict=True)` — c'est le mécanisme
qui a fermé les 34 premières.
