"""Narrer une file de livres, sans surveillance, et se relever tout seul.

Un livre demande trois heures de GPU. Vingt en demandent soixante-cinq, soit
deux jours et demi pendant lesquels personne ne regarde. Ce script existe pour
que cette absence ne coûte rien.

Trois principes, tous appris à la dure :

**Un livre qui échoue n'arrête pas les autres.** Sa cause est notée, il est
marqué en échec, on passe au suivant. Rien n'est plus coûteux qu'une file de
vingt livres bloquée à la troisième heure par le second.

**Rien n'est refait.** Un livre déjà terminé est sauté ; un livre interrompu
reprend au segment près, parce que le cache de ``narrate_book.py`` survit à
tout. Relancer ce script après une coupure ne coûte que ce qui manquait.

**La réparation fait partie du rendu, pas de l'après.** Une fois le livre
narré, les segments jugés fatals — silencieux, tronqués, emballés — sont
re-générés un par un avec un seed dérivé, et leur chapitre est recousu depuis
le cache. Renarrer le chapitre entier pour une phrase serait absurde.

    python scripts/narrate_queue.py queue/queue.json --device cuda
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], logfile: pathlib.Path | None = None) -> tuple[int, str]:
    """Lancer une étape. Sa sortie va dans un fichier, pas en mémoire."""
    if logfile:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
        tail = logfile.read_text(encoding="utf-8", errors="replace")[-600:]
        return p.returncode, tail
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=REPO)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", help="queue.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outroot", default="output")
    ap.add_argument("--qc-retries", default="2")
    ap.add_argument("--only", type=int, help="ne traiter que les N premiers")
    ap.add_argument("--skip-repair", action="store_true")
    args = ap.parse_args()

    qpath = pathlib.Path(args.queue).resolve()
    books = json.loads(qpath.read_text(encoding="utf-8"))
    if args.only:
        books = books[: args.only]

    state_path = qpath.parent / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    def save() -> None:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    logdir = qpath.parent / "logs"
    logdir.mkdir(exist_ok=True)

    log(f"file de {len(books)} livre(s) — {sum(b['chars'] for b in books)} caractères")

    for i, b in enumerate(books, 1):
        slug = b["slug"]
        st = state.get(slug, {})
        if st.get("status") == "done":
            log(f"[{i}/{len(books)}] {slug} : déjà terminé, sauté")
            continue

        txt = qpath.parent / b["txt"]
        outdir = pathlib.Path(args.outroot) / f"book_{slug.replace('-', '_')}"
        blog = logdir / f"{slug}.log"
        state[slug] = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "voice": b["voice"], "chars": b["chars"]}
        save()

        log(f"[{i}/{len(books)}] {slug} — {b['chars']} car., voix « {b['voice']} »")

        # Pré-vol : il ne charge pas le modèle, donc il coûte des secondes et
        # attrape ce qui ferait échouer trois heures plus tard.
        rc, out = run([PYTHON, "scripts/narrate_book.py", str(txt), "--voice", b["voice"],
                       "--device", args.device, "--outdir", str(outdir), "--dry-run"])
        if rc != 0:
            log(f"    pré-vol refusé — livre écarté")
            state[slug].update(status="failed", stage="dry-run", detail=out[-400:])
            save()
            continue

        t0 = time.time()
        rc, tail = run([PYTHON, "scripts/narrate_book.py", str(txt), "--voice", b["voice"],
                        "--device", args.device, "--outdir", str(outdir),
                        "--qc-retries", args.qc_retries,
                        "--assemble", "m4b", "--export-acx"], blog)
        mins = (time.time() - t0) / 60
        if rc != 0:
            log(f"    narration échouée après {mins:.0f} min — voir {blog.name}")
            state[slug].update(status="failed", stage="narration", minutes=round(mins, 1),
                               detail=tail[-400:])
            save()
            continue
        log(f"    narré en {mins:.0f} min")

        # Réparation ciblée : un segment fatal ne justifie pas de refaire son
        # chapitre, encore moins le livre.
        repaired = 0
        if not args.skip_repair:
            rc, out = run([PYTHON, "scripts/repair_segment.py", str(outdir), "--list"])
            # Le marqueur est « FATAL » en capitales (repair_segment.py:68).
            # Chercher « fatal » sans distinction de casse compterait aussi la
            # phrase « Aucun segment fatalement défectueux », donc trouverait un
            # défaut dans tout livre sain et chargerait le modèle pour rien.
            fatal = sum(1 for line in out.splitlines() if "FATAL " in line)
            if fatal:
                log(f"    {fatal} segment(s) fatal(s) — réparation")
                rc, out = run([PYTHON, "scripts/repair_segment.py", str(outdir),
                               "--all-fatal", "--device", args.device], blog)
                repaired = fatal
                if rc != 0:
                    log(f"    réparation incomplète (code {rc})")

        wavs = len(list(outdir.glob("*.wav"))) if outdir.is_dir() else 0
        state[slug].update(status="done", minutes=round(mins, 1), chapters_wav=wavs,
                           repaired=repaired, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        save()
        log(f"    terminé — {wavs} chapitre(s), {repaired} segment(s) réparé(s)")

    done = sum(1 for v in state.values() if v.get("status") == "done")
    failed = [k for k, v in state.items() if v.get("status") == "failed"]
    log(f"file terminée — {done} livre(s) produits, {len(failed)} en échec")
    for k in failed:
        log(f"    échec : {k} ({state[k].get('stage')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
