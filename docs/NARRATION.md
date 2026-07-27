# Guide de narration longue (livres, méditations, podcasts)

Ce guide explique comment utiliser ce fork de VoxCPM2 pour narrer des **contenus
longs** en français : livres audio, scripts de méditation guidée, scripts de
podcast, etc.

## Réponse courte

**Oui, c'est fait pour ça** — mais deux réalités comptent :

1. **La vitesse dépend du matériel.** Sur **GPU CUDA**, c'est rapide et pratique. Sur
   **CPU seul**, c'est ~50× plus lent que le temps réel : utilisable pour des extraits
   courts, impraticable pour un livre entier.
2. **Le découpage est automatique et obligatoire.** Le moteur ne peut pas traiter plus
   de ~8 192 tokens d'un coup — au-delà il s'arrête sur une erreur « KV cache is full »
   (voir `src/voxcpm/model/voxcpm2.py`). L'app et le script découpent le texte en
   phrases pour rester bien en-dessous de cette limite, sans que tu aies à t'en soucier.

## Vitesse : à quoi s'attendre

| Matériel | Vitesse (RTF) | 10 min d'audio | Livre de 3 h |
|---|---|---|---|
| RTX 4090 (CUDA) | ~0.30 (≈3× plus rapide que le réel) | ~3 min | **< 1 h** |
| Apple M4 Pro (Metal) | ~1.76 | ~18 min | ~5 h |
| **CPU seul (cette machine)** | **~50** (50× plus lent) | **~8 h** | **plusieurs jours** |

> RTF = *Real-Time Factor* : temps de calcul ÷ durée audio produite. Plus c'est bas, mieux c'est.

### Activer le GPU (fortement recommandé pour les livres)

1. Installe une version CUDA de PyTorch dans le venv (voir https://pytorch.org — CUDA ≥ 12.0).
2. Lance avec `--device cuda` :
   ```
   .\.venv\Scripts\python.exe app.py --host 127.0.0.1 --port 8808 --device cuda --no-denoiser
   ```
   Le modèle demande ~8 Go de VRAM.

### Astuce vitesse CPU (à tester)

Sur CPU, le modèle tourne actuellement en **bfloat16 émulé**, ce qui est lent
(`pick_runtime_dtype` dans `src/voxcpm/model/utils.py` ne force `float32` que sur MPS,
pas sur CPU). Forcer `float32` sur CPU utiliserait plus de RAM mais serait probablement
plus rapide. C'est une piste d'optimisation non encore intégrée — demande-la si tu veux
qu'on la teste/mesure.

## Deux façons de narrer

### 1. Interface web — pour des extraits / chapitre par chapitre

Idéale pour tester des voix, générer une méditation, un segment de podcast, ou un
chapitre à la fois.

- Charge ton texte avec **« 📄 Charger un fichier .txt »** (ou colle-le).
- Choisis une **voix prédéfinie** (le seed et le style se règlent automatiquement).
- Laisse **« Découper les longs textes »** activé (règle la taille de segment avec le
  curseur si besoin, 100–600 caractères).
- Clique **« Générer la voix »**. Chaque génération est archivée dans `output/` sous un
  nom explicite (`narration_<voix>_seed<seed>_<horodatage>.wav`).

### 2. Script `narrate_book.py` — pour un livre / long script entier

Robuste pour les longs contenus : **sauvegarde par chapitre**, **reprise après
interruption**, **économe en mémoire** (un seul chapitre en RAM à la fois).

Sépare les chapitres de ton `.txt` par une ligne contenant seulement `---` :

```
Chapitre premier. ...

---

Chapitre deuxième. ...
```

Puis :
```
# Aperçu du découpage, sans rien générer :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme" --dry-run

# Génération (une .wav par chapitre dans output/book_<nom>/) :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "Narrateur profond & calme"

# Sur GPU :
.\.venv\Scripts\python.exe scripts\narrate_book.py livre.txt --voice "..." --device cuda
```

- **Reprise** : si le script est interrompu, relance la même commande — les chapitres
  déjà produits sont ignorés (utilise `--force` pour tout régénérer).
- Options utiles : `--chunk-max-chars`, `--silence`, `--cfg`, `--steps`, `--no-normalize`,
  `--chapter-regex` (séparateur de chapitres personnalisé), `--description` + `--seed`
  (voix personnalisée au lieu d'un preset).

## Réglages recommandés par usage

| Usage | Voix suggérée | Réglages |
|---|---|---|
| **Livre audio (fiction)** | *Narrateur profond & calme* / *Narratrice douce & naturelle* | défauts (CFG 2.0, 10 étapes) |
| **Documentaire / non-fiction** | *Narrateur documentaire velouté* / *Narrateur moderne & professionnel* | défauts |
| **Méditation guidée** | *Méditation guidée (grave & lente)* | augmente `--silence` (ex. 0.6–1.0 s) pour de longues pauses |
| **Podcast** | *Conteur jeune & dynamique* / *Narratrice chaleureuse & conversationnelle* | défauts |

## Cohérence de la voix sur un long texte

La voix reste identique d'un segment à l'autre parce que **le même seed est réutilisé
pour tous les segments** (une paire description + seed régénère exactement la même voix).
C'est ce qui garantit un narrateur constant sur tout un livre.

> Note : les jointures entre segments sont de simples silences. Pour des transitions
> encore plus fluides (prosodie enchaînée via *prompt-cache*), une option expérimentale
> serait possible — demande-la si tu en as besoin.

## Limites à connaître

- **Longueur par appel** : ~8 192 tokens max (découpage automatique, donc transparent).
- **Durée par segment** : le moteur vise ~6× la longueur du texte et s'arrête tout seul ;
  garde des segments de taille raisonnable (défaut 300 caractères).
- **Sortie** : WAV 48 kHz. Un livre entier concaténé en un seul fichier serait très
  lourd en mémoire — c'est pourquoi le script écrit **un fichier par chapitre**. Assemble-les
  ensuite avec ton outil audio (ex. `ffmpeg` concat) si tu veux un seul fichier.
