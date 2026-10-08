# Narrer les scripts de MindScript Audio

Le script s'écrit dans [MindScript Audio](https://mindscriptaudio.com) —
méditation, hypnose, relaxation — avec ses silences en repères `[PAUSE:BREATH]`,
`[PAUSE:TRANSITION]`, `[SILENCE 6]`… La narration se fait ici, avec les voix et la
chaîne des livres audio. `scripts/narrate_mindscript.py` fait le trajet dans les
deux sens : il va chercher les scripts, les narre, et peut rapporter le MP3 dans la
bibliothèque audio du compte.

## Entrer dans MindScript

Deux façons, et le script dit laquelle il emploie :

| | Clé d'API | Session |
|---|---|---|
| Comment | `MINDSCRIPT_API_KEY` (créée dans les réglages de l'app) | `MINDSCRIPT_EMAIL` + `MINDSCRIPT_PASSWORD` |
| Qui | comptes **Team / Enterprise** seulement (l'app le vérifie) | tout compte |
| Peut | lire les scripts | lire les scripts, **déposer l'audio** |

`MINDSCRIPT_URL` pointe sur une autre instance (`http://localhost:5000` pour l'app
lancée en local). Rien n'est stocké : les variables d'environnement suffisent.

```powershell
$env:MINDSCRIPT_EMAIL = "..." ; $env:MINDSCRIPT_PASSWORD = "..."
.\.venv\Scripts\python.exe scripts\narrate_mindscript.py --list
```

## Narrer

```bash
# un script, par son id ou le début de son id, sur le GPU loué :
python scripts/narrate_mindscript.py --script 3f2a9c --device cuda

# tout ce qui a changé depuis une date, relu, puis déposé dans l'app :
python scripts/narrate_mindscript.py --all --since 2026-10-01 --relecture --upload

# seulement écrire les .txt (pour narrer à la main ou vérifier le texte) :
python scripts/narrate_mindscript.py --all --export-only
```

Chaque script va dans `output/mindscript/<slug>/` : `script.txt` (le texte tel que
`narrate_book.py` le lit), `script.json` (id, titre, langue, empreinte du texte),
`narration.log`, et le livrable `<slug>_complet.mp3`. La narration est celle de
`narrate_meditations.sh` : un MP3 par script, **sans générique** (une méditation
commence dans le calme, et le disclaimer est déjà dans le texte), repères de
silence respectés, lexique français appliqué, reprise au segment près grâce au cache.

**Ce qui décide de renarrer.** `state.json` garde, par script, l'empreinte de son
texte. Un script inchangé et déjà rendu est **sauté** ; un script **modifié dans
l'app** est renarré — les segments qui n'ont pas bougé viennent du cache, seuls
les autres sont synthétisés, et l'ancien MP3 est effacé pour qu'on ne le confonde
pas avec le nouveau. `--force` renarre quand même. Un script en échec n'arrête pas
les autres ; il est noté dans `state.json` avec son étape.

**La voix.** Par défaut, selon la langue du script : `fr` → « Aurore — méditation
guidée ». Le catalogue n'a pas de voix anglaise : un script en anglais est **écarté
avec sa raison** tant qu'on ne lui donne pas `--voice`. `--voice` vaut pour toute la
passe.

## Rapporter l'audio dans l'app

`--upload` dépose chaque MP3 produit par `POST /api/audio/assets/upload`, sous le
titre du script, catégorie `custom`, étiquettes `mindscript, voxcpm`. Il faut une
session ; une clé d'API ne sait pas déposer, et c'est refusé avant tout appel. Un
dépôt qui échoue ne défait pas la narration : le MP3 est là, l'erreur est dans
`state.json`, et une relance avec `--upload` réessaie.

## Les repères

`narration/cues.py` comprend les repères que MindScript écrit, y compris les formes
des scripts les plus anciens : `[PAUSE]` nu (3 s, comme dans l'app) et `[PAUSE 5]`.
Un repère non reconnu serait **lu à voix haute** — c'est le défaut qu'on ne veut
jamais dans une méditation, et c'est pour ça que chaque forme relevée dans la base
de l'app a son test.

## Limites à connaître

- Le dépôt va dans la **bibliothèque audio** du compte (les « audio assets »), pas
  dans la liste des audios produits par la synthèse de l'app : ce sont deux objets
  différents côté MindScript, et l'API publique ne sait écrire que dans le premier.
- La synthèse se fait là où est VoxCPM, pas dans l'app : sur ce CPU elle est
  quarante fois plus lente que le temps réel ; un script de dix minutes, c'est le
  GPU loué (`docs/CLOUD.md`).
