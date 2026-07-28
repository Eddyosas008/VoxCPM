# Guide de narration longue (livres, méditations, podcasts)

Ce guide explique comment utiliser ce fork de VoxCPM2 pour narrer des **contenus
longs** en français : livres audio, scripts de méditation guidée, scripts de
podcast, etc.

## Réponse courte

**Oui, c'est fait pour ça** — mais deux réalités comptent :

1. **La vitesse dépend du matériel.** Sur **GPU CUDA**, c'est rapide et pratique. Sur
   **CPU seul**, c'est ~40× plus lent que le temps réel : utilisable pour des extraits
   courts, très long pour un livre entier.
2. **Le découpage est automatique et obligatoire.** Le moteur ne peut pas traiter plus
   de ~8 192 tokens d'un coup — au-delà il s'arrête sur une erreur « KV cache is full »
   (voir `src/voxcpm/model/voxcpm2.py`). L'app et le script découpent le texte en
   phrases pour rester bien en-dessous de cette limite, sans que tu aies à t'en soucier.

## La chaîne de production

Le texte brut ne devient pas un livre audio en une étape. Cinq étapes s'enchaînent,
chacune dans un module de `narration/` — testable et utilisable indépendamment :

| Étape | Module | Ce qu'elle fait |
|---|---|---|
| **1. Préparation** | `narration/text_fr.py` | Réécrit le texte tel qu'un narrateur le dirait : `1789` → « mille sept cent quatre-vingt-neuf », `M. Dupont` → « Monsieur Dupont », `XIVe siècle` → « quatorzième siècle », `14h30`, `1 250 €`, `3,5 %`… |
| **2. Découpage** | `narration/chunking.py` | Coupe en segments sous la limite du moteur, **sans jamais couper une phrase**, et décide la durée du silence après chaque segment selon la ponctuation |
| **3. Synthèse** | moteur VoxCPM2 | Même seed partout → voix identique du début à la fin |
| **4. Mastering** | `narration/audio.py` | Rogne les silences parasites, supprime les clics aux jointures, insère les pauses, normalise la sonie **une fois par chapitre** |
| **5. Assemblage** | `narration/assemble.py` | Réunit les chapitres en un seul M4B/MP3 avec marqueurs de chapitres |

Entre les étapes 2 et 3, un **cache par segment** (`narration/cache.py`) rend la
narration reprenable : voir plus bas.

## Vitesse : à quoi s'attendre

| Matériel | Vitesse (RTF) | 10 min d'audio | Livre de 3 h |
|---|---|---|---|
| RTX 4090 (CUDA) | ~0.30 (≈3× plus rapide que le réel) | ~3 min | **< 1 h** |
| Apple M4 Pro (Metal) | ~1.76 | ~18 min | ~5 h |
| **CPU seul (float32, cette machine)** | **~41** (41× plus lent) | **~7 h** | **plusieurs jours** |

> RTF = *Real-Time Factor* : temps de calcul ÷ durée audio produite. Plus c'est bas, mieux c'est.

### Activer le GPU (fortement recommandé pour les livres)

1. Installe une version CUDA de PyTorch dans le venv (voir https://pytorch.org — CUDA ≥ 12.0).
2. Lance avec `--device cuda` :
   ```
   .\.venv\Scripts\python.exe app.py --host 127.0.0.1 --port 8808 --device cuda --no-denoiser
   ```
   Le modèle demande ~8 Go de VRAM.

### Vitesse CPU : float32 par défaut (~1,5× plus rapide)

Sur CPU, le `bfloat16` du checkpoint est **émulé** et lent. Ce fork force donc
`float32` sur CPU (voir `pick_runtime_dtype` dans `src/voxcpm/model/utils.py`), ce qui
est **~1,5× plus rapide** (mesuré : RTF 60,9 → 41,1 sur la même phrase, soit −33 % de
temps) au prix d'un peu plus de RAM.

Pour revenir à l'ancien comportement bfloat16 : `set VOXCPM_CPU_DTYPE=bfloat16` avant de
lancer l'app (ou export sous bash).

## Trois façons de narrer

### 1. Onglet « 📚 Livre audio » — pour un livre depuis l'interface

1. Choisis d'abord une voix dans l'onglet **🎙️ Studio** (la description et le seed
   de cette voix sont ceux qui seront utilisés).
2. Passe sur l'onglet **📚 Livre audio**, charge ton `.txt` ou colle le texte.
3. Clique **« 🔍 Analyser sans générer »** : tu vois le nombre de chapitres, de
   segments, la durée estimée, et **le premier segment tel qu'il sera réellement lu**
   (après préparation du texte). C'est le moment de repérer un nombre ou une
   abréviation mal interprétés — avant d'engager des heures de calcul.
4. Clique **« 📖 Narrer le livre »**. Chaque chapitre terminé est écrit sur disque
   et devient écoutable immédiatement ; l'avancement s'affiche au fur et à mesure.
5. Clique **« 📦 Assembler le livre audio »** pour obtenir un fichier unique.

### 2. Script `narrate_book.py` — pour un livre entier en ligne de commande

Le plus robuste pour les longs contenus.

Sépare les chapitres de ton `.txt` par une ligne contenant seulement `---` :

```
Chapitre premier. ...

---

Chapitre deuxième. ...
```

Puis :
```
# Aperçu : découpage, durée estimée, et texte préparé — sans rien générer :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme" --dry-run

# Génération (une .wav par chapitre dans output/book_<nom>/) :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme"

# Génération + assemblage direct en M4B avec chapitres :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme" ^
    --assemble m4b --title "Mon Livre" --author "Edwin"

# Sur GPU :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "..." --device cuda
```

### 3. Onglet « 🎙️ Studio » — pour des extraits

Idéal pour tester des voix, générer une méditation ou un segment de podcast.
Deux options utiles dans les **Réglages avancés** :

- **Préparation du texte français** — applique l'étape 1 de la chaîne.
- **Mastering livre audio** — applique l'étape 4 (activé par défaut).

## Reprise après interruption

C'est le point critique sur CPU, où un chapitre prend des heures.

- **Par chapitre** : un chapitre dont le `.wav` existe déjà est ignoré. `--force` le
  régénère.
- **Par segment** : chaque segment généré est mis en cache dans
  `output/book_<nom>/.cache/`, indexé par le **contenu** (texte + description + seed +
  CFG + étapes + modèle). Si tu relances après une interruption, seuls les segments
  manquants sont calculés — pas tout le chapitre.

Conséquences pratiques :

- Corriger une coquille dans un paragraphe n'invalide que les segments de ce
  paragraphe. Le reste du livre est réutilisé tel quel.
- Changer de voix (ou de seed) invalide tout, ce qui est correct : c'est un autre
  narrateur.
- Le cache occupe de la place. `--no-cache` le désactive ; supprimer le dossier
  `.cache/` est sans risque une fois le livre terminé.

## Qualité audio : la norme ACX

Les plateformes de livres audio vérifient trois choses. Le mastering vise ces valeurs,
et chaque chapitre est mesuré à l'écriture :

| Mesure | Cible | Pourquoi |
|---|---|---|
| RMS (sonie) | entre **-23 et -18 dBFS** (défaut : -20) | volume homogène d'un chapitre à l'autre |
| Crête | **≤ -3 dBFS** | marge avant saturation |
| Bruit de fond | **≤ -60 dBFS** | silences réellement silencieux |

Le RMS est mesuré **sur la parole seule** : les silences entre phrases sont exclus du
calcul. Sans cela, un chapitre aux pauses généreuses mesurerait plusieurs dB trop bas,
et le corriger pousserait la parole au-dessus du plafond de crête.

Vérifier un livre assemblé :
```
.\.venv\Scripts\python.exe scripts\assemble_audiobook.py output\book_mon_livre --check
```

## Contrôle qualité automatique

Le moteur échoue rarement, mais il échoue **localement** : un segment sur quelques
dizaines revient coupé au milieu d'un mot, muet, ou parti en boucle bien après la fin
de son texte. Sur GPU on régénère le chapitre. Sur CPU un chapitre représente des
heures, donc la seule réparation abordable porte sur le segment fautif — encore
faut-il le trouver. Écouter quatre heures de narration pour repérer onze secondes
n'est pas une méthode.

Chaque segment généré est donc confronté **au texte qui l'a produit**. C'est ce
couplage qui rend la détection possible : l'audio seul ne peut pas dire si deux
secondes constituent une phrase complète, mais deux secondes pour deux cents
caractères sont une troncature, sans ambiguïté.

| Code | Gravité | Ce qui est détecté |
|---|---|---|
| `silent` | fatal | rien n'est revenu |
| `truncated` | fatal | beaucoup moins d'audio que le texte ne l'implique |
| `runaway` | fatal | beaucoup plus — le moteur a bouclé ou divagué |
| `clipped` | fatal | échantillons saturés, irrécupérables au mastering |
| `gap` | suspect | long silence interne, signature d'une proposition sautée |
| `abrupt_end` | suspect | s'arrête au niveau de parole, sans décroissance |
| `looped` | suspect | l'enveloppe de niveau se répète |

Seuls les défauts **fatals** déclenchent une régénération, avec une seed dérivée de
la seed d'origine — donc reproductible : le même livre relancé de zéro répare le même
segment de la même façon. Le meilleur essai est conservé, jamais le dernier : un
second tirage peut être pire que le premier, et garder silencieusement le pire
rendrait la réparation nuisible.

```bash
# Comportement par défaut : un nouvel essai par segment fatalement défectueux
python scripts/narrate_book.py livre.txt --voice "..."

# Plus insistant sur un livre qu'on ne veut pas réécouter segment par segment
python scripts/narrate_book.py livre.txt --voice "..." --qc-retries 3

# Signaler sans régénérer (utile pour auditer un livre déjà produit)
python scripts/narrate_book.py livre.txt --voice "..." --qc-retries 0

# Sortie en code d'erreur s'il reste un défaut — pour un enchaînement automatisé
python scripts/narrate_book.py livre.txt --voice "..." --qc-strict
```

Le bilan est écrit dans `output/book_<nom>/qc_report.json` : un segment par entrée,
avec sa durée, son débit et ses défauts. `--no-qc` désactive tout.

**Sur les seuils de débit.** Ce sont eux qui portent la détection de troncature, et
ils viennent de la mesure, pas d'une estimation : sur la même phrase de 81
caractères, les sept voix préréglées produisent entre **15,8 et 24,1 caractères par
seconde**, soit plus de 50 % d'écart entre la plus lente et la plus rapide. Les
bornes (35 et 6 car/s) sont donc placées largement en dehors de cette plage — choisir
une voix rapide ne doit jamais ressembler à un défaut — tout en restant franchies par
une troncature qui perdrait la moitié d'une phrase. Un test verrouille ces valeurs
mesurées, pour qu'un réglage ultérieur ne puisse pas les faire dériver sans alerte.

## Assemblage en un fichier unique

```
# M4B avec marqueurs de chapitres (nécessite ffmpeg) :
.\.venv\Scripts\python.exe scripts\assemble_audiobook.py output\book_mon_livre ^
    --title "Mon Livre" --author "Edwin" --format m4b
```

**ffmpeg n'est pas installé sur cette machine.** Ce n'est pas bloquant : le script
produit quand même le WAV complet et le fichier de marqueurs, puis affiche la commande
exacte à lancer une fois ffmpeg installé. Les heures de synthèse ne sont jamais perdues
à cause d'un encodeur manquant.

Les titres de chapitres viennent, dans l'ordre : de `--titles`, puis d'un fichier
`titles.txt` à côté des WAV (écrit automatiquement par `narrate_book.py` à partir de la
première ligne de chaque chapitre), puis des noms de fichiers.

## Prononciation : lexique personnalisé

`conf/pronunciation_fr.json` associe ce qui est écrit à ce qui doit être prononcé.
C'est l'outil pour les noms propres d'un roman, les sigles et les mots étrangers :

```json
{
  "SNCF": "S N C F",
  "Nietzsche": "Nitche",
  "Aurélien Krähenbühl": "Aurélien Krènebul"
}
```

Le remplacement est insensible à la casse et ne s'applique qu'à des mots entiers.
Les clés commençant par `_` sont des commentaires.

## Ce que la préparation du texte corrige (et ses limites)

Sont gérés : nombres cardinaux et ordinaux (`1er`, `2e`, `1re`), décimales, sommes en
euros/dollars/livres, pourcentages, heures (`14h30`), abréviations (`M.`, `Mme`, `Dr`,
`Me`, `St`, `etc.`, `av. J.-C.`, `n°`, `p. 42`), chiffres romains, tirets de dialogue,
guillemets, et le balisage Markdown.

Les chiffres romains ne sont développés que dans des contextes **non ambigus** :
après un mot déclencheur (`chapitre XIV`, `tome III`), en forme ordinale (`XIXe`), ou
seuls sur une ligne de titre. C'est délibéré : « Le » est L + e, « Ce » est C + e — les
développer partout ferait lire « Le manuscrit » comme « cinquantième manuscrit ».

Désactiver globalement : `--no-text-prep` (script) ou décocher la case (interface).

## Réglages recommandés par usage

| Usage | Voix suggérée | Réglages |
|---|---|---|
| **Livre audio (fiction)** | *Narrateur profond & calme* / *Narratrice douce & naturelle* | défauts (CFG 2.0, 10 étapes) |
| **Documentaire / non-fiction** | *Narrateur documentaire velouté* / *Narrateur moderne & professionnel* | défauts |
| **Méditation guidée** | *Méditation guidée (grave & lente)* | `--pause-sentence 0.8 --pause-paragraph 1.6` |
| **Podcast** | *Conteur jeune & dynamique* / *Narratrice chaleureuse & conversationnelle* | `--pause-paragraph 0.6` (rythme plus soutenu) |

## Cohérence de la voix sur un long texte

La voix reste identique d'un segment à l'autre parce que **le même seed est réutilisé
pour tous les segments** (une paire description + seed régénère exactement la même voix).
C'est ce qui garantit un narrateur constant sur tout un livre.

L'option expérimentale `--continuity` va plus loin : chaque segment est enchaîné à
partir du précédent (continuation par *prompt-cache*) pour des jointures encore plus
fluides. Le mécanisme fonctionne mais il est **beaucoup plus lent** — à régler sur GPU.

## Limites à connaître

- **Longueur par appel** : ~8 192 tokens max (découpage automatique, donc transparent).
- **Durée par segment** : le moteur vise ~6× la longueur du texte et s'arrête tout seul ;
  garde des segments de taille raisonnable (défaut 300 caractères).
- **Sortie** : les chapitres sont écrits en WAV 16 bits, un fichier par chapitre. Un
  livre entier n'est jamais chargé en mémoire — ni à la génération, ni à l'assemblage
  (qui écrit en flux).
