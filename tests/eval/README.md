# Évaluation

Deux harnais, pour deux questions différentes.

## `test_selection_eval.py` — la sélection propose-t-elle le bon outil ?

Tourne en CI : aucun modèle, aucun réseau, seulement les cas annotés de
`cases.py`. Mesure le **rappel** — un outil qui n'est pas proposé ne peut pas
être choisi, alors que la précision ne coûte que du contexte.

```bash
pytest tests/eval
```

Les seuils sont des **planchers**, pas des objectifs : ils existent pour
attraper une régression, et sont placés un peu sous ce que l'implémentation
actuelle atteint.

## `run_tool_calling.py` — le modèle appelle-t-il le bon outil ?

Exige un serveur Ollama et un modèle téléchargé, donc **hors CI**. À lancer à
la main au moment de choisir un modèle ou de régler la sélection.

```bash
python tests/eval/run_tool_calling.py --model llama3.1:8b
python tests/eval/run_tool_calling.py --model qwen3:8b --compare
```

`--compare` exécute les deux variantes — avec la sélection top-k et avec tous
les outils — et affiche l'écart. C'est la mesure qui dit si la sélection vaut
sa complexité **sur ta machine**.

## Ajouter des cas

Écris-les comme quelqu'un les dirait à voix haute. Un cas qui reprend les mots
de la description de l'outil ne mesure rien.

Les deux langues comptent : les descriptions d'outils sont en anglais et
l'assistant répond dans la langue qu'on lui parle. C'est précisément là que le
rapprochement lexical échoue, et c'est ce que `TERM_ALIASES` dans
`text_utils.py` compense — imparfaitement, et seulement pour le vocabulaire
qui y figure.
