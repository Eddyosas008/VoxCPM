"""Les repères de mise en voix venus des scripts MindScript.

Un script écrit dans MindScript Audio porte des repères entre crochets, de
deux natures que tout oppose : des silences mesurés — ``[PAUSE:BREATH]``,
``[SILENCE 6]`` — et des indications d'interprétation — ``[RALENTIR]``,
``[EMPHASE]``, ``[TON CHALEUREUX]``. Le moteur ne sait pas « ralentir » à la
demande, mais il sait se taire : les premiers deviennent des durées exactes
dans le plan de montage, les seconds sont retirés avant la synthèse. Un
repère lu à voix haute est pire qu'un repère ignoré.

Les durées sont celles que l'app applique elle-même à l'écoute
(``shared/studio/narration-document.ts``) : BREATH quatre secondes,
TRANSITION cinq, INTEGRATION huit, DEEP dix, et trois pour un repère de
pause inconnu — le script a voulu un silence, il en obtient un plausible
plutôt qu'aucun.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

__all__ = [
    "CUE_RE",
    "DEFAULT_PAUSE_SECONDS",
    "PAUSE_SECONDS",
    "cue_seconds",
    "shield",
    "split_on_cues",
    "strip_cues",
    "unshield",
]

PAUSE_SECONDS = {
    "BREATH": 4.0,
    "TRANSITION": 5.0,
    "INTEGRATION": 8.0,
    "DEEP": 10.0,
}

DEFAULT_PAUSE_SECONDS = 3.0

#: Les formes que MindScript écrit réellement. ``TON`` avale son complément
#: (« TON PLUS GRAVE ») ; ``SILENCE`` accepte une virgule décimale et un « s »
#: d'unité, parce que les scripts sont écrits à la main. ``[PAUSE]`` nu et
#: ``[PAUSE 5]`` sont les formes des scripts générés avant les repères nommés
#: — l'app leur donne trois secondes et la durée demandée ; un ``[PAUSE]``
#: qui ne serait pas reconnu serait lu « pause » au milieu d'une méditation.
CUE_RE = re.compile(
    r"\[\s*(?:"
    r"PAUSE\s*:\s*(?P<pause>[A-ZÀ-Ü]+)"
    r"|(?P<bare>PAUSE)(?:\s+(?P<pause_seconds>\d+(?:[.,]\d+)?)(?:\s*S)?)?"
    r"|SILENCE\s+(?P<seconds>\d+(?:[.,]\d+)?)(?:\s*S)?"
    r"|(?P<breath>RESPIRATION)"
    r"|(?P<direction>RALENTIR|EMPHASE|TON\s[^\]]*)"
    r")\s*\]",
    re.IGNORECASE,
)

_PLACEHOLDER = "\x00cue{index}\x00"
_PLACEHOLDER_RE = re.compile("\x00cue(\\d+)\x00")


def cue_seconds(match: "re.Match[str]") -> Optional[float]:
    """La durée d'un repère, ou ``None`` pour une indication sans silence."""
    if match.group("direction") is not None:
        return None
    if match.group("seconds") is not None:
        return float(match.group("seconds").replace(",", "."))
    if match.group("breath") is not None:
        return PAUSE_SECONDS["BREATH"]
    if match.group("bare") is not None:
        asked = match.group("pause_seconds")
        return float(asked.replace(",", ".")) if asked else DEFAULT_PAUSE_SECONDS
    return PAUSE_SECONDS.get(match.group("pause").upper(), DEFAULT_PAUSE_SECONDS)


def shield(text: str) -> Tuple[str, List[str]]:
    """Remplace chaque repère par un jeton que la préparation ne touche pas.

    La préparation du texte épelle les nombres — ``[SILENCE 6]`` deviendrait
    ``[SILENCE six]`` et ne serait plus un repère. Le jeton utilise un octet
    nul, que ni la typographie ni les abréviations ne réécrivent.
    """
    saved: List[str] = []

    def keep(match: "re.Match[str]") -> str:
        saved.append(match.group(0))
        return _PLACEHOLDER.format(index=len(saved) - 1)

    return CUE_RE.sub(keep, text), saved


def unshield(text: str, saved: List[str]) -> str:
    """L'inverse de :func:`shield`, après la préparation."""
    return _PLACEHOLDER_RE.sub(lambda m: saved[int(m.group(1))], text)


def strip_cues(text: str) -> str:
    """Le texte sans aucun repère — ce que le moteur doit prononcer."""
    return re.sub(r"\s{2,}", " ", CUE_RE.sub(" ", text)).strip()


def split_on_cues(text: str) -> List[Tuple[str, Optional[float]]]:
    """Découpe sur les repères : ``[(texte, silence forcé après lui), …]``.

    Le silence d'un passage est la somme des repères de pause qui le suivent
    immédiatement — deux repères consécutifs sont deux silences que l'app
    aurait rendus l'un après l'autre. ``None`` veut dire : aucun repère ici,
    le plan de pauses ordinaire décide.
    """
    # Les indications d'interprétation ne coupent rien : retirées d'abord,
    # pour qu'un [EMPHASE] au milieu d'une phrase ne la scinde pas en deux.
    text = CUE_RE.sub(
        lambda m: m.group(0) if cue_seconds(m) is not None else " ", text
    )

    runs: List[Tuple[str, Optional[float]]] = []
    position = 0

    for match in CUE_RE.finditer(text):
        before = text[position : match.start()].strip()
        position = match.end()
        seconds = cue_seconds(match)
        if before:
            runs.append((before, seconds))
        elif runs:
            # Repère sans texte neuf devant lui : il prolonge le silence du
            # passage précédent — deux repères d'affilée font deux silences.
            previous_text, previous_pause = runs[-1]
            runs[-1] = (previous_text, (previous_pause or 0.0) + seconds)
        # Avant tout texte, un silence n'a rien à faire taire : ignoré.

    tail = text[position:].strip()
    if tail:
        runs.append((tail, None))
    return runs
