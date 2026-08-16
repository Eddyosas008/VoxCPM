#!/usr/bin/env python3
"""Construit la file de narration à partir du catalogue de manuscrits.

Un livre entre dans la file quand quatre choses existent : un manuscrit que
le préparateur accepte, un titre, un auteur, une couverture. Tout le reste a
un défaut raisonnable. Ce qui manque est dit, pas deviné : un livre incomplet
est écarté avec sa raison, parce qu'un livre qui échoue à la minute quatre-
vingt-dix-huit coûte plus cher que celui qu'on n'a pas lancé.

Les métadonnées ont deux âges dans ce catalogue. Les livres récents portent
un ``metadata/book_metadata.json`` ; les plus anciens n'ont que leur propre
page de titre, en tête du manuscrit — un ``#`` pour le titre, un ``##`` pour
le sous-titre, un nom en gras pour l'auteur. Les deux sont lus.

La voix se choisit sur le sujet, pas sur l'auteur : c'est la règle que suivent
déjà les vingt et un livres narrés. Un propos intime, parental ou
thérapeutique va à la voix féminine ; un essai, une enquête ou un atlas va à
la voix masculine. Le champ ``why`` garde la raison, pour qu'un choix puisse
être discuté plus tard au lieu d'être subi.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import subprocess
import sys

VOIX_FEMININE = "Aurore — livre audio"
VOIX_MASCULINE = "Alex Somerset"

IMAGES = (".jpg", ".jpeg", ".png", ".webp")

#: Côté minimal d'une couverture audio, en pixels : c'est le seuil d'Audible
#: (ACX), et il est repris tel quel par les autres distributeurs.
COTE_MINIMAL = 2400

#: Un sujet intime, parental ou thérapeutique appelle la voix féminine.
#:
#: Attention à la portée de cette règle : confrontée aux vingt et un livres
#: déjà attribués à la main, elle en contredit quatre. Elle ne sert donc
#: qu'aux livres encore sans voix — un choix éditorial déjà fait ne se
#: recalcule pas, il se conserve (cf. ``--file-existante``).
#:
#: Les alternances courtes sont bornées par \b : sans cela « non » se trouve
#: au milieu de n'importe quel mot et emporte le classement.
INTIME = re.compile(
    r"m[ée]dit|respir|sommeil|dormir|anxi[ée]t|d[ée]press|trauma|couple|"
    r"enfant|parent|maternit|m[ée]nopause|intimit|int[ée]rieur|apais|"
    r"calme|matins|habitudes|renouer|estrangement|panique|f[ée]minin|"
    r"tdah|th[ée]rap|\bsoi\b|\bflow\b|\bnon\b|\bvoix\b|\bpiliers\b|"
    r"\bcorps\b|\bfemmes?\b|\bintention\b",
    re.IGNORECASE,
)


def dimensions(f: pathlib.Path) -> tuple[int, int] | None:
    """Largeur et hauteur d'une image, lues dans son en-tête.

    Sans dépendance : le catalogue tient dans trois formats et leurs en-têtes
    tiennent en trente lignes. Rien n'est décodé, seuls les premiers octets
    sont lus, donc mesurer cent couvertures coûte le prix d'un ``ls``.
    """
    try:
        with f.open("rb") as fh:
            tete = fh.read(32)
            if tete[:8] == b"\x89PNG\r\n\x1a\n":
                l, h = struct.unpack(">II", tete[16:24])
                return int(l), int(h)
            if tete[:4] == b"RIFF" and tete[8:12] == b"WEBP":
                fh.seek(0)
                d = fh.read(40)
                if d[12:16] == b"VP8X":
                    return (int.from_bytes(d[24:27], "little") + 1,
                            int.from_bytes(d[27:30], "little") + 1)
                if d[12:16] == b"VP8 ":
                    return (int.from_bytes(d[26:28], "little") & 0x3FFF,
                            int.from_bytes(d[28:30], "little") & 0x3FFF)
                if d[12:16] == b"VP8L":
                    b = int.from_bytes(d[21:25], "little")
                    return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
                return None
            if tete[:2] == b"\xff\xd8":
                # JPEG : sauter de marqueur en marqueur jusqu'au SOFn, seul
                # segment qui porte les dimensions. Les SOF 4, 8 et 12 sont
                # des marqueurs de table, pas des cadres — d'où l'exclusion.
                fh.seek(2)
                while True:
                    octet = fh.read(1)
                    if not octet:
                        return None
                    if octet != b"\xff":
                        continue
                    while octet == b"\xff":
                        octet = fh.read(1)
                    marqueur = octet[0]
                    if 0xC0 <= marqueur <= 0xCF and marqueur not in (0xC4, 0xC8, 0xCC):
                        fh.read(3)
                        h, l = struct.unpack(">HH", fh.read(4))
                        return int(l), int(h)
                    taille = struct.unpack(">H", fh.read(2))[0]
                    fh.seek(taille - 2, 1)
    except (OSError, struct.error, IndexError):
        return None
    return None


def couverture(d: pathlib.Path) -> pathlib.Path | None:
    """La couverture *audio* : carrée, et d'au moins 2400 pixels de côté.

    Un dossier de livre contient plusieurs couvertures qui ne servent pas au
    même produit — la jaquette imprimée, le rabat complet, la vignette ebook
    en portrait. Prendre la première venue passe inaperçu jusqu'au dépôt, où
    Audible refuse tout ce qui n'est pas carré ; la contrainte du distributeur
    est donc devenue la règle de choix, au lieu d'un ordre de dossiers qui ne
    la connaissait pas. À égalité, le fichier nommé pour l'audio l'emporte,
    puis le plus grand.

    Une couverture ebook 1600×2560 avait ainsi été retenue pour trois livres,
    embarquée dans leur M4B, et n'aurait été rejetée qu'au dépôt.
    """
    vues: list[pathlib.Path] = []
    for sous in ("_covers_v3", "_covers_v2", "_covers", "couverture", "formats"):
        rep = d / sous
        if rep.is_dir():
            vues += sorted(rep.rglob("*"))
    vues += sorted(d.glob("*"))

    candidates = [f for f in vues
                  if f.suffix.lower() in IMAGES and f.is_file()
                  and f.stat().st_size > 20_000]

    # Classer avant de mesurer, et s'arrêter à la première conforme. Le
    # catalogue vit sur OneDrive, où lire le moindre octet d'un fichier le
    # fait descendre en entier : mesurer les dix images d'un livre pour n'en
    # garder qu'une rapatriait des gigaoctets et prenait des dizaines de
    # minutes. L'ordre reflète la préférence — le fichier nommé pour l'audio,
    # puis le plus grand — donc le résultat est celui du meilleur candidat,
    # pas celui du premier rencontré.
    candidates.sort(key=lambda f: (0 if "audio" in f.name.lower() else 1,
                                   -f.stat().st_size))
    for f in candidates:
        dim = dimensions(f)
        if dim is not None and dim[0] == dim[1] and dim[0] >= COTE_MINIMAL:
            return f
    return None


def _depuis_json(d: pathlib.Path):
    bm = d / "metadata" / "book_metadata.json"
    if not bm.is_file():
        return None
    try:
        m = json.loads(bm.read_text(encoding="utf-8")).get("book_metadata", {})
    except (json.JSONDecodeError, OSError):
        return None
    if not m.get("titre"):
        return None
    return m["titre"], m.get("sous_titre", ""), m.get("auteur", "")


def _depuis_manuscrit(man: pathlib.Path):
    """La page de titre du manuscrit, quand aucun fichier ne la porte.

    On ne lit que la tête : au-delà, un ``#`` est un titre de chapitre et
    non le titre du livre. « Front matter » est un intitulé de section, pas
    un titre — le vrai suit.
    """
    tete = man.read_text(encoding="utf-8", errors="replace")[:4000]
    titres = re.findall(r"(?m)^#\s+(.+?)\s*$", tete)
    titres = [t for t in titres if t.strip().lower() not in ("front matter", "sommaire")]
    if not titres:
        return None
    sous = re.search(r"(?m)^##\s+(.+?)\s*$", tete)
    sous_t = sous.group(1).strip() if sous else ""
    if sous_t.lower() in ("sommaire", "table des matières"):
        sous_t = ""
    auteur = re.search(r"(?m)^\*\*([^*]{3,60})\*\*\s*$", tete)
    return titres[0].strip(), sous_t, (auteur.group(1).strip() if auteur else "")


def metadonnees(d: pathlib.Path, man: pathlib.Path):
    return _depuis_json(d) or _depuis_manuscrit(man) or (None, "", "")


def choisir_voix(slug: str, titre: str, sous_titre: str) -> tuple[str, str]:
    matiere = f"{slug} {titre} {sous_titre}"
    if INTIME.search(matiere):
        return VOIX_FEMININE, "sujet intime ou d'accompagnement — voix féminine"
    return VOIX_MASCULINE, "essai ou enquête — registre documentaire"


def main() -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogue", help="dossier contenant les livre-*/")
    ap.add_argument("-o", "--output", default="queue/queue.json")
    ap.add_argument("--txt-dir", default="queue", help="où écrire les textes préparés")
    ap.add_argument("--covers-dir", default="queue/covers")
    ap.add_argument("--min-chars", type=int, default=50_000)
    ap.add_argument("--file-existante", metavar="JSON",
                    help="file déjà curée : ses voix, titres et auteurs sont "
                         "conservés tels quels. Un choix éditorial déjà fait "
                         "ne se recalcule pas.")
    ap.add_argument("--dry-run", action="store_true",
                    help="ne rien écrire : dire seulement ce qui entrerait")
    args = ap.parse_args()

    ancienne: dict[str, dict] = {}
    if args.file_existante:
        p = pathlib.Path(args.file_existante)
        if p.is_file():
            q = json.loads(p.read_text(encoding="utf-8"))
            for it in (q if isinstance(q, list) else q.get("books", [])):
                ancienne[it["slug"]] = it
            print(f"file existante : {len(ancienne)} livres déjà curés, conservés\n")

    racine = pathlib.Path(args.catalogue)
    if not racine.is_dir():
        print(f"catalogue introuvable : {racine}", file=sys.stderr)
        return 1

    txt_dir = pathlib.Path(args.txt_dir)
    cov_dir = pathlib.Path(args.covers_dir)
    if not args.dry_run:
        txt_dir.mkdir(parents=True, exist_ok=True)
        cov_dir.mkdir(parents=True, exist_ok=True)

    prepare = pathlib.Path(__file__).with_name("prepare_manuscript.py")
    retenus, ecartes = [], []

    for d in sorted(racine.iterdir()):
        if not d.is_dir():
            continue
        man = d / "manuscrit_complet.md"
        if not man.is_file():
            continue
        if man.stat().st_size < args.min_chars:
            ecartes.append((d.name, f"manuscrit trop court ({man.stat().st_size} o)"))
            continue

        titre, sous_titre, auteur = metadonnees(d, man)
        if not titre:
            ecartes.append((d.name, "aucun titre trouvé"))
            continue
        cov = couverture(d)
        if cov is None:
            ecartes.append((d.name, f"aucune couverture carrée d'au moins "
                                    f"{COTE_MINIMAL} px"))
            continue

        txt = txt_dir / f"{d.name}.txt"
        if not args.dry_run:
            r = subprocess.run(
                [sys.executable, str(prepare), str(man), "-o", str(txt)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                ecartes.append((d.name, f"préparation refusée : {r.stderr.strip()[:60]}"))
                continue
            corps = txt.read_text(encoding="utf-8")
            chapitres = corps.count("\n---\n") + 1
            chars = len(corps)
            cible = cov_dir / f"{d.name}{cov.suffix.lower()}"
            cible.write_bytes(cov.read_bytes())
            cov_rel = f"{cov_dir.as_posix()}/{cible.name}"
        else:
            chapitres, chars, cov_rel = 0, man.stat().st_size, str(cov)

        # Un livre déjà curé garde sa voix, son titre et son auteur : ces
        # choix ont été faits à l'oreille et à la lecture, la règle ci-dessous
        # n'en sait pas autant.
        vieux = ancienne.get(d.name)
        if vieux:
            voix = vieux.get("voice", VOIX_FEMININE)
            pourquoi = vieux.get("why", "")
            titre = vieux.get("title") or titre
            sous_titre = vieux.get("subtitle", sous_titre)
            auteur = vieux.get("author") or auteur
        else:
            voix, pourquoi = choisir_voix(d.name, titre, sous_titre)

        retenus.append({
            "slug": d.name,
            "txt": f"{d.name}.txt",
            "voice": voix,
            "why": pourquoi,
            "cloned": True,
            "chapters": chapitres,
            "chars": chars,
            "title": titre,
            "subtitle": sous_titre,
            "author": auteur or "Edwin Osayamwen",
            "renarration": False,
            "cover": cov_rel,
            "nouveau": vieux is None,
        })

    # Le plus long d'abord : un échec de disque ou de GPU arrive alors sur le
    # livre le plus coûteux, quand la marge est encore intacte.
    retenus.sort(key=lambda b: -b["chars"])

    print(f"retenus : {len(retenus)}    écartés : {len(ecartes)}\n")
    par_voix: dict[str, int] = {}
    for b in retenus:
        par_voix[b["voice"]] = par_voix.get(b["voice"], 0) + 1
        marque = "NEUF " if b["nouveau"] else "     "
        print(f"  {marque}{b['slug']:<36}{b['chars']:>8,} · "
              f"{b['voice'][:20]:<22}{b['title'][:36]}".replace(",", " "))
    if ecartes:
        print("\nécartés :")
        for slug, raison in ecartes:
            print(f"  {slug:<36}{raison}")

    total = sum(b["chars"] for b in retenus)
    print(f"\nrépartition des voix : {par_voix}")
    print(f"total : {total:,} caractères".replace(",", " "))
    print(f"estimation : {total / 250_000 * 2.75:.0f} h de GPU, "
          f"{total / 250_000 * 2.75 * 0.34:.0f} $, "
          f"{total / 250_000 * 450 / 1024:.1f} Go de livrables")

    if args.dry_run:
        print("\n(--dry-run : rien n'a été écrit)")
        return 0

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(retenus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfile écrite : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
