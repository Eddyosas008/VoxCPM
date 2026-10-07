"""Faire relire un livre narré par la reconnaissance vocale, et réparer ce qui s'arrête en route.

Le contrôle qualité ne voit pas une phrase coupée en deux : son débit est
normal, sa fin est propre. Mesuré sur un livre narré, environ 2 % des segments
s'arrêtent avant la fin de leur texte, et aucun seuil acoustique ne les sépare
des sains. Ce script fait transcrire chaque segment du cache par Whisper et
regarde jusqu'où dans le texte la transcription va (voir ``narration.relecture``).

Avec ``--repair``, chaque segment tronqué est régénéré avec une graine dérivée,
relu, et gardé s'il va cette fois jusqu'au bout — sinon on réessaie, et au
pire on garde la prise qui va le plus loin. Les chapitres touchés sont
reconstruits depuis le cache ; rien d'autre n'est synthétisé.

Il écrit ``relecture_report.json`` à côté des chapitres. La file le lit pour
savoir si le M4B et l'export ACX doivent être refaits.

Exemples
--------
  # Relire tout le livre (GPU : quelques minutes pour trois heures d'audio) :
  python scripts/relire_livre.py output/book_mon_livre --device cuda

  # Relire et réparer :
  python scripts/relire_livre.py output/book_mon_livre --device cuda --repair

  # Un échantillon, pour se faire une idée sur CPU :
  python scripts/relire_livre.py output/book_mon_livre --device cpu --sample 40

Codes de sortie : 0 quand aucun segment tronqué ne reste ; 1 quand il en reste
(réparé ou non, c'est dit) ; 2 quand le livre ne peut pas être relu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narration import cache as cache_tools  # noqa: E402
from narration import relecture, repair  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="dossier d'un livre narré (plan.json et .cache)")
    ap.add_argument("--device", default="cuda", help="cuda, cpu (défaut : cuda)")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="ne relire que N segments répartis dans le livre (0 = tous)")
    ap.add_argument("--portee", type=float, default=relecture.Seuils.portee_min,
                    help="portée alignée en deçà de laquelle un segment est tronqué "
                         f"(défaut : {relecture.Seuils.portee_min})")
    ap.add_argument("--repair", action="store_true",
                    help="régénérer les segments tronqués et reconstruire leurs chapitres")
    ap.add_argument("--retries", type=int, default=2,
                    help="essais par segment tronqué avec --repair (défaut : 2)")
    ap.add_argument("--model-id", default="openbmb/VoxCPM2", help="modèle de synthèse")
    ap.add_argument("--whisper", default=relecture.MODELE, help="modèle de reconnaissance")
    ap.add_argument("--json", metavar="FICHIER",
                    help=f"écrire le rapport ici plutôt que dans {relecture.RAPPORT}")
    return ap


def _labels(plan: repair.BookPlan) -> List[str]:
    return [repair.segment_label(c.index, p)
            for c in plan.chapters for p in range(1, len(c.segments) + 1)]


def _echantillon(labels: List[str], n: int) -> Optional[List[str]]:
    """N labels répartis régulièrement, ou None pour « tous »."""
    if n <= 0 or n >= len(labels):
        return None
    pas = max(1, len(labels) // n)
    return labels[::pas][:n]


def reparer(plan: repair.BookPlan, cache: cache_tools.ChunkCache, demo, transcrire,
            verdict: relecture.Relecture, seuils: relecture.Seuils, retries: int,
            ) -> Tuple[bool, relecture.Relecture]:
    """Réessayer un segment tronqué jusqu'à ce qu'il aille au bout.

    Rend ``(réparé, meilleur verdict)``. Quand aucun essai ne va au bout, la
    prise qui va le plus loin reste en cache — ce peut être l'originale.
    """
    chapter_index, position = repair.parse_label(verdict.label)
    segment = plan.segment(chapter_index, position)
    spec = plan.voice_spec()
    key = cache.key(segment.text, spec)
    render = repair.renderer(demo, plan, segment.text)

    original = cache.get(key)
    candidats: List[Tuple[relecture.Relecture, int, "object", int]] = []
    if original is not None:
        candidats.append((verdict, original[0], original[1], cache.attempt_of(key)))

    for _ in range(max(1, retries)):
        result = repair.reroll_segment(plan, chapter_index, position, cache, render,
                                       keep_worse=False)
        if not result.improved:
            # Le contrôle qualité l'a rejetée (muette, coupée net...) : elle
            # n'est pas en cache, et pas la peine de la relire.
            print(f"    essai {result.attempt} : rejeté par le contrôle qualité "
                  f"({result.report.describe()})")
            continue
        relu = relecture.juger(verdict.label, segment.text,
                               transcrire(result.sample_rate, result.wav), seuils)
        print(f"    essai {result.attempt} : {relu.describe()}")
        candidats.append((relu, result.sample_rate, result.wav, result.attempt))
        if not relu.tronque:
            return True, relu

    # Rien n'est allé au bout : garder ce qui va le plus loin, et le dire.
    meilleur = max(candidats, key=lambda c: c[0].portee) if candidats else None
    if meilleur is not None:
        relu, sample_rate, wav, attempt = meilleur
        cache.put(key, sample_rate, wav, text=segment.text, attempt=attempt)
        return False, relu
    return False, verdict


def main() -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args()
    outdir = Path(args.directory)
    try:
        plan = repair.BookPlan.load(outdir)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    cache = cache_tools.ChunkCache(outdir / ".cache")
    if not any(cache.root.glob("*.wav")):
        print(f"aucun cache de segments dans {cache.root} — la relecture lit le cache, "
              "elle doit passer avant le balayage (--keep deliverables).", file=sys.stderr)
        return 2

    seuils = relecture.Seuils(portee_min=args.portee)
    labels = _labels(plan)
    cibles = _echantillon(labels, args.sample)
    print(f"Livre       : {outdir}")
    print(f"Segments    : {len(labels)} planifiés, "
          f"{len(cibles) if cibles is not None else len(labels)} à relire")
    print(f"Relecture   : {args.whisper} sur {args.device}", flush=True)

    transcrire = relecture.transcripteur_whisper(args.device, args.whisper)

    def progres(fait: int, total: int) -> None:
        if fait % 50 == 0 or fait == total:
            print(f"    {fait}/{total} segments relus", flush=True)

    verdicts = relecture.relire_livre(plan, cache, transcrire, seuils, cibles, progres)
    if not verdicts:
        # « 0 relu, 0 tronqué » ressemble à s'y méprendre à un livre sain. Ce
        # n'en est pas un : le plan et le cache ne se répondent pas (voix ou
        # modèle changés depuis, cache d'une autre prise). Le dire, et sortir
        # autrement que « rien à signaler ».
        _ecrire_rapport(outdir, args.json, verdicts, [])
        print(f"\naucun des {len(labels)} segments planifiés n'a été retrouvé dans le cache : "
              "plan.json et .cache ne correspondent pas (voix, modèle ou référence changés "
              "depuis la narration ?). Relancez narrate_book.py avec les mêmes arguments "
              "pour les remettre d'accord.", file=sys.stderr)
        return 2
    tronques = [v for v in verdicts if v.tronque]
    print(f"\n{len(verdicts)} segment(s) relu(s), {len(tronques)} tronqué(s)")
    for v in tronques:
        print(f"  {v.label}  {v.describe()}")
        print(f"      demandé : « {v.source[:100]}{'…' if len(v.source) > 100 else ''} »")
        print(f"      entendu : « {v.entendu[:100]}{'…' if len(v.entendu) > 100 else ''} »")

    repares: List[str] = []
    finals: Dict[str, relecture.Relecture] = {v.label: v for v in verdicts}
    if args.repair and tronques:
        try:
            repair.renderer(None, plan, "")
        except repair.ReferenceUnavailable as error:
            print(f"\nRéparation impossible : {error}", file=sys.stderr)
            _ecrire_rapport(outdir, args.json, verdicts, repares)
            return 1

        import app  # noqa: E402 - torch, seulement maintenant

        demo = app.VoxCPMDemo(model_id=args.model_id, device=args.device, load_denoiser=False)
        touches = set()
        for v in tronques:
            print(f"\n{v.label} — réparation")
            ok, final = reparer(plan, cache, demo, transcrire, v, seuils, args.retries)
            finals[v.label] = final
            if ok:
                repares.append(v.label)
            touches.add(repair.parse_label(v.label)[0])
            if not ok:
                print(f"    aucun essai ne va au bout — gardé : {final.describe()}")
        for chapter_index in sorted(touches):
            rebuilt = repair.rebuild_chapter(plan, chapter_index, cache, outdir)
            if rebuilt.ok:
                print(f"Chapitre {chapter_index:03d} reconstruit -> {rebuilt.path.name}")
            else:
                print(f"Chapitre {chapter_index:03d} NON reconstruit : "
                      f"{len(rebuilt.missing)} segment(s) absent(s) du cache")

    rapport = _ecrire_rapport(outdir, args.json, verdicts, repares)
    restants = rapport["restants"]
    if tronques:
        print(f"\n{len(repares)} réparé(s), {len(restants)} restant(s) à l'oreille"
              + (f" : {', '.join(restants)}" if restants else ""))
    return 1 if restants else 0


def _ecrire_rapport(outdir: Path, chemin: Optional[str], verdicts, repares) -> dict:
    rapport = relecture.resume(verdicts, repares)
    cible = Path(chemin) if chemin else outdir / relecture.RAPPORT
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rapport : {cible}")
    return rapport


if __name__ == "__main__":
    raise SystemExit(main())
