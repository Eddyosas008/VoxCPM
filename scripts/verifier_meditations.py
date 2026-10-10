"""Vérifier une série de scripts de méditation avant de la narrer.

Un script de méditation se juge à la lecture, mais une série de cinquante ne
se relit pas en entier avant chaque narration. Ce script attrape ce qui coûte
des heures de GPU ou une séance ratée : la durée hors cible, un repère mal
écrit (lu à voix haute), un chiffre (épelé de travers), une parenthèse (tronque
la phrase), une phrase qui commence par « Expirez » (la voix en coupe la fin),
du markdown, la mention finale absente. Il mesure comme la chaîne mesure : les
repères avec ``narration.cues``, les segments avec ``narration.chunking``.

    python scripts/verifier_meditations.py "C:/.../serie-02-anxiete" --minutes 15

Sort 0 si tout passe, 1 sinon, en nommant chaque fichier et chaque défaut.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from narration import cues  # noqa: E402

#: Débit de lecture retenu pour estimer la durée : la voix de méditation lit
#: lentement, cent trente mots par minute mesurés sur la série 01.
MOTS_PAR_MINUTE = 130.0

MENTION = "ne remplace pas l'avis d'un professionnel"


@dataclass
class Verdict:
    fichier: pathlib.Path
    mots: int = 0
    silence_s: float = 0.0
    minutes: float = 0.0
    defauts: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.defauts


def verifier(fichier: pathlib.Path, minutes_cible: float, tolerance: float) -> Verdict:
    v = Verdict(fichier)
    try:
        texte = fichier.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        v.defauts.append("pas en UTF-8")
        return v

    # Repères : tout ce qui est entre crochets doit être un repère connu.
    for crochet in re.findall(r"\[[^\]]*\]", texte):
        m = cues.CUE_RE.fullmatch(crochet)
        if m is None:
            v.defauts.append(f"repère inconnu, serait lu à voix haute : {crochet}")
        elif cues.cue_seconds(m) is None:
            v.defauts.append(f"indication d'interprétation inutile : {crochet}")
    v.silence_s = sum(s for _, s in cues.split_on_cues(texte) if s)
    prononce = cues.strip_cues(texte)
    v.mots = len(re.findall(r"[A-Za-zÀ-ÿ'’-]+", prononce))
    v.minutes = v.mots / MOTS_PAR_MINUTE + v.silence_s / 60.0

    if abs(v.minutes - minutes_cible) > tolerance:
        v.defauts.append(f"durée estimée {v.minutes:.1f} min, cible {minutes_cible:.0f} ± {tolerance:.0f}")
    if re.search(r"\d", prononce):
        v.defauts.append("chiffre dans le texte prononcé (à écrire en lettres)")
    if re.search(r"[()]", prononce):
        v.defauts.append("parenthèse dans le texte prononcé (tronque la phrase)")
    for m in re.finditer(r"(?:^|[.!?…]\s+)(Expirez)", prononce, re.M):
        v.defauts.append("phrase commençant par « Expirez » (la voix en coupe la fin)")
        break
    if re.search(r"^\s*[#*]|^\s*-\s", texte, re.M) or "**" in texte:
        v.defauts.append("markdown dans le texte")
    if re.search(r"[;—]", prononce):
        v.defauts.append("point-virgule ou tiret cadratin")
    if MENTION not in prononce.lower():
        v.defauts.append("mention finale absente")
    if re.search(r"\b(etc|ex|cf)\.", prononce, re.I):
        v.defauts.append("abréviation")
    return v


def main() -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dossier")
    ap.add_argument("--minutes", type=float, default=15.0, help="durée cible (défaut : 15)")
    ap.add_argument("--tolerance", type=float, default=1.5, help="écart admis en minutes (défaut : 1,5)")
    ap.add_argument("--attendus", type=int, default=0, help="nombre de fichiers attendus (0 = pas de contrôle)")
    args = ap.parse_args()

    dossier = pathlib.Path(args.dossier)
    fichiers = sorted(dossier.glob("*.txt"))
    if args.attendus and len(fichiers) != args.attendus:
        print(f"{len(fichiers)} fichier(s) .txt, {args.attendus} attendus")
    verdicts = [verifier(f, args.minutes, args.tolerance) for f in fichiers]

    print(f"{'fichier':<52} {'mots':>5} {'silence':>8} {'min':>5}  défauts")
    for v in verdicts:
        etat = "ok" if v.ok else " ; ".join(v.defauts)
        print(f"{v.fichier.name[:52]:<52} {v.mots:>5} {v.silence_s:>7.0f}s {v.minutes:>5.1f}  {etat}")
    fautifs = [v for v in verdicts if not v.ok]
    total = sum(v.minutes for v in verdicts)
    print(f"\n{len(verdicts)} script(s), {len(fautifs)} à corriger, ~{total/60:.1f} h d'audio au total")
    return 1 if fautifs or (args.attendus and len(fichiers) != args.attendus) else 0


if __name__ == "__main__":
    raise SystemExit(main())
