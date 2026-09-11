# Suite de tests Jarvis

```bash
pip install -r ../requirements-test.txt
pytest                                  # ~1,2 s
pytest --cov --cov-report=term-missing
pytest tests/test_llm_module.py -v      # un module
pytest -k "whitelist"                   # par mot-clé
pytest -rx                              # détail des xfail (= bugs connus)
```

## Aucune dépendance native requise

`sounddevice`, `webrtcvad`, `faster-whisper` et `ollama` sont remplacés par des
doublures installées dans `sys.modules` par `conftest.py` avant l'import des
modules Jarvis. Pas de PortAudio, pas de compilateur C, pas de téléchargement
de modèle, pas de serveur Ollama.

Les doublures de `_stubs.py` sont **fidèles, pas permissives** : `FakeVad`
rejette les trames que le vrai `webrtcvad` rejette, et `SubscriptableModel`
n'est pas sérialisable en JSON comme les modèles Pydantic du vrai client
ollama. C'est ce qui permet à la suite de mettre BUG-18 et BUG-08 en évidence
plutôt que de les masquer.

## Bugs connus = tests `xfail(strict=True)`

Chaque anomalie de [`../AUDIT.md`](../AUDIT.md) a un test marqué
`xfail(strict=True)` portant son identifiant `BUG-xx`.

- Tant que le bug existe, le test échoue → `xfail` → la suite reste verte.
- Dès que le bug est corrigé, le test réussit → `XPASS` → **la suite échoue**,
  ce qui force à retirer le marqueur dans le même commit que le correctif.

Corriger un bug se fait donc en trois gestes : écrire le correctif, retirer
le `@pytest.mark.xfail`, vérifier que le test passe.

`pytest -rx` liste les 38 marqueurs en cours avec leur motif.

## Fixtures principales (`conftest.py`)

| Fixture | Rôle |
|---|---|
| `settings` | Le singleton de configuration, restauré après chaque test |
| `sandbox` | Répertoire temporaire déclaré comme seul `ALLOWED_DIRECTORIES` |
| `executor` | `ActionExecutor` sandboxé, whitelist isolée sur disque |
| `approve_all` / `approve_and_whitelist` / `deny_all` | Rappels de confirmation, avec journal des appels |
| `fake_sd` / `fake_vad` / `fake_whisper` / `fake_ollama` | Les modules doublures, réinitialisés entre chaque test |
| `clean_env` | Retire toutes les variables Jarvis pour observer les défauts déclarés |

## Conventions

- Un test = une affirmation. Les noms décrivent le comportement attendu, pas
  la méthode appelée.
- Aucun test n'écrit dans le dépôt : tout passe par `tmp_path`.
- Les tests d'intégration (`test_integration.py`) utilisent les vrais
  `LLMModule`, `ActionExecutor` et `Jarvis` ; seules les frontières de
  processus sont simulées.
