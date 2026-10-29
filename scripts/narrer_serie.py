"""Narrer une série de scripts courts (méditations, séances) sans surveillance.

Un ``.txt`` = un MP3. Une séance déjà rendue est sautée ; une séance qui
échoue est retentée (le chargement du modèle meurt parfois faute de mémoire,
sans rien d'abîmé : un second essai, la mémoire revenue, passe) puis laissée
pour la suivante — une séance en échec n'arrête jamais la série. Le cache de
``narrate_book.py`` fait reprendre une séance interrompue au segment près.

Une séance rendue passe ensuite par ``repair_segment.py --codes`` : les
segments à fin coupée (``abrupt_end``, défaut que le contrôle laisse passer
comme simple suspicion) sont régénérés, puis la séance est réassemblée depuis
le cache. ``--reparer ""`` coupe cette étape.

Chaque MP3 fini est recopié à côté de son script, là où on l'écoute.

    python scripts/narrer_serie.py "C:/.../serie-02-anxiete" --device cpu

Le titre d'une séance vient de ``THEMES.md`` s'il existe dans le dossier
(lignes ``NN Titre — …`` ou ``NN Titre. Outil : …``), sinon du nom de fichier.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def titres_du_dossier(dossier: pathlib.Path) -> Dict[str, str]:
    """``{"01": "Le matin anxieux", …}`` depuis THEMES.md, s'il existe."""
    chemin = dossier / "THEMES.md"
    titres: Dict[str, str] = {}
    if not chemin.is_file():
        return titres
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\d\d)\s+(.+?)(?:\s+—\s|\.\s+Outil\s*:|$)", ligne.strip())
        if m:
            titres[m.group(1)] = m.group(2).strip().rstrip(".")
    return titres


def titre_depuis_slug(slug: str) -> str:
    texte = re.sub(r"^\d+_", "", slug).replace("_", " ").strip()
    return texte[:1].upper() + texte[1:] if texte else slug


def livrable(outdir: pathlib.Path) -> Optional[pathlib.Path]:
    trouves = sorted(outdir.glob("*_complet.mp3")) if outdir.is_dir() else []
    return trouves[0] if trouves else None


def commande(txt: pathlib.Path, outdir: pathlib.Path, titre: str, args) -> List[str]:
    cmd = [sys.executable, str(REPO / "scripts" / "narrate_book.py"), str(txt),
           "--voice", args.voice, "--title", titre,
           "--no-credits", "--assemble", "mp3",
           "--outdir", str(outdir), "--device", args.device,
           "--qc-retries", str(args.qc_retries)]
    cmd += ["--cover", args.cover] if args.cover else ["--no-cover"]
    for lex in args.lexicon:
        cmd += ["--lexicon", lex]
    # Tout argument inconnu ici passe tel quel à narrate_book : --cfg, --steps,
    # --pause-sentence, --room-tone… sans tout redéclarer.
    return cmd + list(getattr(args, "reglages", []))


def reparer(outdir: pathlib.Path, journal: pathlib.Path, args, env) -> bool:
    """Régénère les segments signalés par ``args.reparer`` ; vrai si un a changé."""
    cmd = [sys.executable, str(REPO / "scripts" / "repair_segment.py"), str(outdir),
           "--codes", args.reparer, "--device", args.device]
    sortie = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=REPO, env=env, text=True, encoding="utf-8", errors="replace")
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== réparation {args.reparer}\n{sortie.stdout or ''}")
    return "reconstruit" in (sortie.stdout or "")


def main() -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dossier", help="dossier des scripts .txt")
    ap.add_argument("--voice", default="Aurore — méditation guidée")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outroot", default=str(REPO / "output" / "meditations"))
    ap.add_argument("--lexicon", action="append", default=None)
    ap.add_argument("--qc-retries", default="2")
    ap.add_argument("--cover")
    ap.add_argument("--essais", type=int, default=3,
                    help="essais par séance avant de passer à la suivante (défaut : 3)")
    ap.add_argument("--attente", type=float, default=60.0,
                    help="secondes d'attente entre deux essais (défaut : 60)")
    ap.add_argument("--reparer", default="abrupt_end",
                    help="défauts à régénérer après la narration (défaut : abrupt_end ; \"\" : aucun)")
    ap.add_argument("--only", type=int, help="ne traiter que les N premières séances restantes")
    ap.add_argument("--pas-de-copie", action="store_true",
                    help="ne pas recopier les MP3 à côté des scripts")
    args, args.reglages = ap.parse_known_args()
    if args.lexicon is None:
        args.lexicon = [str(REPO / "conf" / "pronunciation_fr.json")]

    dossier = pathlib.Path(args.dossier)
    scripts = sorted(dossier.glob("*.txt"))
    scripts = [s for s in scripts if not s.name.upper().startswith("LISEZ")]
    if not scripts:
        log(f"aucun script .txt dans {dossier}")
        return 1
    titres = titres_du_dossier(dossier)
    serie = dossier.name
    outroot = pathlib.Path(args.outroot) / serie
    outroot.mkdir(parents=True, exist_ok=True)
    etat_path = outroot / "state.json"
    etat = json.loads(etat_path.read_text(encoding="utf-8")) if etat_path.is_file() else {}

    def sauver() -> None:
        etat_path.write_text(json.dumps(etat, ensure_ascii=False, indent=2), encoding="utf-8")

    restants = [s for s in scripts if livrable(outroot / s.stem) is None]
    if args.only:
        restants = restants[: args.only]
    log(f"série {serie} : {len(scripts)} script(s), {len(scripts) - len(restants) if not args.only else '?'} "
        f"déjà rendu(s), {len(restants)} à narrer — voix « {args.voice} », {args.device}")

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    ok = echecs = 0
    for i, txt in enumerate(restants, 1):
        slug = txt.stem
        outdir = outroot / slug
        titre = titres.get(slug[:2], titre_depuis_slug(slug))
        journal = outroot / f"{slug}.log"
        cmd = commande(txt, outdir, titre, args)
        log(f"[{i}/{len(restants)}] {slug} — « {titre} »")
        etat[slug] = {"status": "running", "title": titre,
                      "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        sauver()
        t0 = time.time()
        rc = None
        for essai in range(1, max(1, args.essais) + 1):
            with journal.open("a", encoding="utf-8") as fh:
                fh.write(f"\n=== essai {essai} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.flush()
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=REPO, env=env).returncode
            if rc == 0 and livrable(outdir) is not None:
                break
            log(f"    essai {essai} en échec (code {rc})"
                + (f" — nouvel essai dans {args.attente:.0f} s" if essai < args.essais else ""))
            if essai < args.essais:
                time.sleep(args.attente)
        mp3 = livrable(outdir)
        minutes = (time.time() - t0) / 60
        if rc != 0 or mp3 is None:
            echecs += 1
            etat[slug].update(status="failed", code=rc, minutes=round(minutes, 1))
            sauver()
            log(f"    ÉCHEC après {minutes:.0f} min — voir {journal.name}")
            continue
        if args.reparer and reparer(outdir, journal, args, env):
            # Les chapitres ont changé : la même commande réassemble le MP3
            # depuis le cache, sans rien synthétiser.
            with journal.open("a", encoding="utf-8") as fh:
                fh.write("\n=== réassemblage\n")
                fh.flush()
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=REPO, env=env).returncode
            mp3 = livrable(outdir) or mp3
            log("    segments réparés, séance réassemblée"
                + ("" if rc == 0 else f" (réassemblage en échec, code {rc} : MP3 d'avant gardé)"))
        if not args.pas_de_copie:
            shutil.copy2(mp3, dossier / mp3.name)
        ok += 1
        etat[slug].update(status="done", mp3=str(mp3), minutes=round(minutes, 1),
                          finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        sauver()
        log(f"    terminé en {minutes:.0f} min -> {mp3.name}")

    log(f"fin de passe — {ok} rendue(s), {echecs} échec(s)")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
