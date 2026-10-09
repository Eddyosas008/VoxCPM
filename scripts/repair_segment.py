"""Find and repair a single defective segment of an already narrated book.

A book is nine hours of CPU. When one sentence in it comes out truncated, or
babbling, or silent, re-narrating the book — or even the chapter — to fix that
sentence is absurd: everything else was fine, and the cache still holds it.

This re-generates **one segment** with a derived seed and stitches its chapter
back together from the cache. Nothing else is synthesized.

It needs the ``plan.json`` written beside the chapters, which records which
cached entry holds which sentence. Books narrated before that file existed can
be repaired by re-running ``narrate_book.py`` with the same arguments: the
segments all come from the cache, so it costs seconds and writes the plan.

Examples
--------
  # What is wrong, worst first:
  ./.venv/Scripts/python.exe scripts/repair_segment.py output/book_mon_livre --list

  # Repair one, then rebuild its chapter:
  ./.venv/Scripts/python.exe scripts/repair_segment.py output/book_mon_livre --segment ch003/seg012

  # Not happy with the new take? Ask again — a different one comes out:
  ./.venv/Scripts/python.exe scripts/repair_segment.py output/book_mon_livre --segment ch003/seg012

  # Repair every fatally defective segment in one pass:
  ./.venv/Scripts/python.exe scripts/repair_segment.py output/book_mon_livre --all-fatal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
from narration import cache as cache_tools  # noqa: E402
from narration import modele, quality, repair  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", help="Directory holding the chapters and plan.json")

    what = parser.add_argument_group("quoi réparer")
    what.add_argument("--list", action="store_true",
                      help="List the defective segments and exit")
    what.add_argument("--segment", metavar="LABEL",
                      help="Segment to re-generate, e.g. ch003/seg012")
    what.add_argument("--all-fatal", action="store_true",
                      help="Repair every fatally defective segment")
    what.add_argument("--attempt", type=int,
                      help="Force an attempt number, so a given take is reproducible")
    what.add_argument("--keep-worse", action="store_true",
                      help="Keep the new take even when it is worse than the old one")

    run = parser.add_argument_group("exécution")
    run.add_argument("--device", default="cpu", help="auto, cpu, mps, cuda (default: cpu)")
    run.add_argument("--model-id", default="openbmb/VoxCPM2", help="Model path or HF repo id")
    return parser


def describe(label: str, report: quality.SegmentReport) -> str:
    kind = "FATAL " if report.fatal else "suspect"
    return f"  {label}  {kind}  {report.describe()}"


def main() -> int:
    args = build_parser().parse_args()

    outdir = Path(args.directory)
    try:
        plan = repair.BookPlan.load(outdir)
    except FileNotFoundError:
        raise SystemExit(
            f"No {repair.PLAN_FILENAME} in {outdir}. Re-run narrate_book.py with the same "
            "arguments to write one — every segment comes from the cache, so it costs seconds."
        )
    except ValueError as error:
        raise SystemExit(str(error))

    cache = cache_tools.ChunkCache(outdir / ".cache")
    reports = repair.inspect_book(plan, cache)
    if not reports:
        raise SystemExit(
            f"No cached segment found in {outdir / '.cache'}. Nothing can be repaired without "
            "the cache the narration wrote."
        )

    flagged = repair.flagged_segments(reports)
    print(f"Livre       : {outdir}")
    print(f"Segments    : {len(reports)} en cache · {len(flagged)} signalé(s)")
    for label, report in flagged:
        print(describe(label, report))
    if not flagged:
        print("  (aucun défaut détecté)")

    if args.list or (not args.segment and not args.all_fatal):
        if not args.list:
            print("\nRien à faire : précisez --segment LABEL ou --all-fatal.")
        return 0

    targets: List[str]
    if args.all_fatal:
        targets = [label for label, report in flagged if report.fatal]
        if not targets:
            print("\nAucun segment fatalement défectueux : rien à régénérer.")
            return 0
    else:
        targets = [args.segment]

    # The model is loaded once, and only now: listing defects reads the cache
    # and must not cost a minute of model load.
    # A cloned voice is checked before the model loads: refusing after a minute
    # of model load would be the same refusal, later.
    try:
        repair.renderer(None, plan, "")
    except repair.ReferenceUnavailable as error:
        raise SystemExit(str(error))

    resolu = modele.resoudre(args.model_id)
    print(f"Modèle      : {args.model_id} — {resolu.describe()}")
    demo = app.VoxCPMDemo(model_id=resolu.chemin, device=args.device, load_denoiser=False)

    touched_chapters = set()
    failures = 0
    for label in targets:
        try:
            chapter_index, position = repair.parse_label(label)
            segment = plan.segment(chapter_index, position)
        except (ValueError, KeyError) as error:
            print(f"\n{label} : {error}")
            failures += 1
            continue

        print(f"\n{label} — « {segment.text[:90]}{'…' if len(segment.text) > 90 else ''} »")
        result = repair.reroll_segment(
            plan,
            chapter_index,
            position,
            cache,
            # Rendered the way the book was: same description, same reference
            # recording for a cloned voice. The old inline render passed no
            # reference at all, which would have spoken the repaired sentence
            # in another voice.
            repair.renderer(demo, plan, segment.text),
            attempt=args.attempt,
            keep_worse=args.keep_worse,
        )
        print(f"  essai {result.attempt}, graine {result.seed}")
        print(f"  avant : {result.previous.describe() if result.previous else '(rien en cache)'}")
        print(f"  après : {result.report.describe()}")
        if not result.improved:
            # Saying so matters: the user would otherwise believe the repair
            # took, and hear the same defect on the next listen.
            print("  le nouvel essai est moins bon, l'ancien est conservé — relancez pour "
                  "en tirer un autre")
            failures += 1
            continue
        touched_chapters.add(chapter_index)

    for chapter_index in sorted(touched_chapters):
        rebuilt = repair.rebuild_chapter(plan, chapter_index, cache, outdir)
        if rebuilt.ok:
            print(f"\nChapitre {chapter_index:03d} reconstruit -> {rebuilt.path.name}")
        else:
            print(f"\nChapitre {chapter_index:03d} NON reconstruit : "
                  f"{len(rebuilt.missing)} segment(s) absent(s) du cache")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
