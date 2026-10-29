#!/usr/bin/env python3
"""Pose un fond sonore apaisant sous une méditation narrée.

Le fond est synthétisé ici, pas tiré d'une banque de sons : aucune licence à
vérifier, et il dure exactement ce que dure la séance. Deux ambiances :

* ``nappe``  — un accord grave tenu (ré, la, ré, fa dièse), chaque note
  doublée d'une jumelle à peine désaccordée qui la fait respirer, et dont le
  volume ondule lentement, sans rythme repérable ;
* ``vagues`` — un bruit brun filtré qui monte et retombe comme une mer
  lointaine, à des intervalles irréguliers de sept à treize secondes.

Le fond s'efface de quelques décibels quand la voix parle et revient dans les
silences, puis continue seul après la dernière phrase avant de s'éteindre :
pour une séance de sommeil, la fin ne doit pas tomber d'un coup.

    python scripts/fond_sonore.py seance.mp3 seance_fond.mp3 --ambiance nappe+vagues
    python scripts/fond_sonore.py dossier/ --suffixe _fond      # tous les *_complet.mp3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100


def lire(path: Path, tempo: float) -> np.ndarray:
    filtre = ["-af", f"atempo={tempo}"] if abs(tempo - 1.0) > 1e-3 else []
    brut = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *filtre, "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(brut, dtype=np.float32).copy()


def ecrire(path: Path, stereo: np.ndarray, bitrate: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
         "-c:a", "libmp3lame", "-b:a", bitrate, str(path)],
        input=np.ascontiguousarray(stereo, dtype=np.float32).tobytes(), check=True,
    )


def passe_bas(x: np.ndarray, coupure: float) -> np.ndarray:
    """Un pôle simple, appliqué deux fois : doux, sans résonance."""
    from scipy.signal import lfilter

    a = np.exp(-2 * np.pi * coupure / SR)
    for _ in range(2):
        x = lfilter([1 - a], [1, -a], x)
    return x


def lent(n: int, rng: np.random.Generator, periode: tuple[float, float]) -> np.ndarray:
    """Une ondulation lente entre 0 et 1, sans période fixe."""
    t = np.arange(n) / SR
    env = np.zeros(n)
    for _ in range(3):
        p = rng.uniform(*periode)
        env += np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
    return 0.5 + env / 6


def nappe(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    notes = [(73.42, 1.0), (110.0, 0.8), (146.83, 0.55), (185.0, 0.3), (220.0, 0.25)]
    canaux = []
    for _ in range(2):
        s = np.zeros(n)
        for f, a in notes:
            for d in (-0.11, 0.13):
                ton = np.sin(2 * np.pi * (f + d) * t + rng.uniform(0, 2 * np.pi))
                ton += 0.12 * np.sin(2 * np.pi * 2 * (f + d) * t)
                s += a * ton * (0.35 + 0.65 * lent(n, rng, (18, 45)))
        canaux.append(passe_bas(s, 900))
    return np.stack(canaux, axis=1)


def vagues(n: int, rng: np.random.Generator) -> np.ndarray:
    # L'enveloppe : des houles de sept à treize secondes, de hauteurs inégales.
    env = np.zeros(n)
    pos = 0
    while pos < n:
        duree = int(rng.uniform(7, 13) * SR)
        x = np.linspace(0, 1, duree)
        houle = np.sin(np.pi * x) ** 2 * rng.uniform(0.55, 1.0)
        fin = min(n, pos + duree)
        env[pos:fin] = np.maximum(env[pos:fin], houle[: fin - pos])
        pos += int(duree * rng.uniform(0.55, 0.8))
    canaux = []
    for decalage in (0, int(0.35 * SR)):
        bruit = np.cumsum(rng.standard_normal(n))            # bruit brun
        bruit -= passe_bas(bruit, 8)                          # sans dérive
        grave = passe_bas(bruit, 500)
        ecume = passe_bas(rng.standard_normal(n), 2500) * 0.02
        e = np.roll(env, decalage)
        canaux.append(grave * (0.25 + 0.75 * e) + ecume * e ** 2)
    return np.stack(canaux, axis=1)


def sons(n: int, dossier: Path, rng: np.random.Generator, fondu: float = 6.0) -> np.ndarray:
    """Les fichiers d'un dossier, dans un ordre tiré au sort, enchaînés en
    fondu jusqu'à couvrir la séance : on ne repère pas de boucle."""
    fichiers = sorted(p for p in dossier.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac"))
    if not fichiers:
        raise SystemExit(f"aucun son dans {dossier}")
    f = int(fondu * SR)
    rampe = np.linspace(0, 1, f)[:, None]
    sortie = np.zeros((0, 2))
    while sortie.shape[0] < n:
        for p in rng.permutation(len(fichiers)):
            brut = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(fichiers[p]), "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                check=True, capture_output=True,
            ).stdout
            x = np.frombuffer(brut, dtype=np.float32).reshape(-1, 2).astype(np.float64)
            x /= np.sqrt(np.mean(np.square(x))) + 1e-12
            if sortie.shape[0] >= f and x.shape[0] > 2 * f:
                x[:f] *= rampe
                sortie[-f:] *= rampe[::-1]
                sortie[-f:] += x[:f]
                x = x[f:]
            sortie = np.concatenate([sortie, x])
            if sortie.shape[0] >= n:
                break
    return sortie[:n]


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def poser(voix: np.ndarray, ambiance: str, niveau: float, effacement: float,
          traine: float, graine: int, dossier: Path | None = None) -> np.ndarray:
    n = voix.size + int(traine * SR)
    rng = np.random.default_rng(graine)
    fond = sons(n, dossier, rng) if dossier else np.zeros((n, 2))
    for nom, poids in (("nappe", 1.0), ("vagues", 0.8)):
        if nom in ambiance and dossier is None:
            couche = nappe(n, rng) if nom == "nappe" else vagues(n, rng)
            couche /= np.sqrt(np.mean(np.square(couche))) + 1e-12
            fond += poids * couche
    fond *= 10 ** ((niveau - rms_db(fond)) / 20)

    # L'effacement suit la voix : attaque d'un tiers de seconde, retour en
    # deux secondes, pour que le fond ne pompe pas à chaque mot.
    v = np.zeros(n)
    v[: voix.size] = np.abs(voix)
    parle = (passe_bas(v, 3) > 10 ** (-42 / 20)).astype(float)
    from scipy.signal import lfilter

    a_mont, a_desc = np.exp(-1 / (0.35 * SR)), np.exp(-1 / (2.0 * SR))
    # Deux passes d'un pôle : montée rapide puis descente lente, approchées
    # par le maximum des deux lissages.
    lisse = np.maximum(lfilter([1 - a_mont], [1, -a_mont], parle),
                       lfilter([1 - a_desc], [1, -a_desc], parle))
    gain = 10 ** (-effacement * np.clip(lisse, 0, 1) / 20)
    fond *= gain[:, None]

    # Entrée en huit secondes, extinction sur les trois quarts de la traîne.
    entree = min(n, 8 * SR)
    fond[:entree] *= np.linspace(0, 1, entree)[:, None] ** 2
    sortie = int(traine * 0.75 * SR)
    if sortie:
        fond[-sortie:] *= np.linspace(1, 0, sortie)[:, None] ** 2

    mix = fond.copy()
    mix[: voix.size] += voix[:, None]
    crete = np.max(np.abs(mix))
    if crete > 0.97:
        mix *= 0.97 / crete
    return mix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entree", type=Path, help="un MP3, ou un dossier où chercher les *_complet.mp3")
    ap.add_argument("sortie", type=Path, nargs="?", help="MP3 de sortie (fichier unique)")
    ap.add_argument("--ambiance", default="nappe+vagues", help="nappe, vagues ou nappe+vagues")
    ap.add_argument("--sons", type=Path, help="dossier de sons à enchaîner à la place de l'ambiance synthétisée")
    ap.add_argument("--niveau", type=float, default=-33.0, help="niveau du fond en dBFS RMS (défaut -33)")
    ap.add_argument("--effacement", type=float, default=5.0, help="dB retirés au fond sous la voix (défaut 5)")
    ap.add_argument("--traine", type=float, default=45.0, help="secondes de fond après la voix (défaut 45)")
    ap.add_argument("--tempo", type=float, default=1.0, help="vitesse de la voix, 0.94 = 6 %% plus lente")
    ap.add_argument("--suffixe", default="_fond", help="suffixe des fichiers produits en mode dossier")
    ap.add_argument("--bitrate", default="192k")
    args = ap.parse_args(argv)

    if args.entree.is_dir():
        travaux = [(p, p.with_name(p.stem + args.suffixe + ".mp3"))
                   for p in sorted(args.entree.rglob("*_complet.mp3"))]
    else:
        travaux = [(args.entree, args.sortie or args.entree.with_name(args.entree.stem + args.suffixe + ".mp3"))]
    for i, (src, dst) in enumerate(travaux):
        voix = lire(src, args.tempo)
        mix = poser(voix, args.ambiance, args.niveau, args.effacement, args.traine, graine=i,
                    dossier=args.sons)
        ecrire(dst, mix, args.bitrate)
        print(f"{dst.name} : {mix.shape[0] / SR / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
