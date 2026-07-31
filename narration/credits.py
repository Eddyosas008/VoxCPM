"""Opening and closing credits, the way distributors require them.

A finished audiobook is not just the book read aloud. Every distributor — ACX
and Audible, and behind them Amazon, Apple Books, Kobo, Google Play — requires
the recording to *announce itself*: the first file opens on the title, the
author and the narrator, and the last one closes by naming them again. A
submission without them is rejected at quality review before anyone listens to
a word of the prose.

The rules this module encodes:

* **Opening credits** carry title, subtitle when there is one, author, narrator.
  They are the very first thing heard.
* **Closing credits** name the work and its author again, then the narrator, and
  may carry production and rights information.
* **Synthetic narration is disclosed.** Where no human narrator is named, the
  credit says the reading is a synthetic voice. Audible distributes such titles
  through a separate programme and labels them; claiming a machine reading as a
  human performance is what gets an account closed, so the disclosure is the
  default and switching it off has to be a deliberate act.

Only the *text* lives here. It is narrated by the same voice, with the same
seed, through the same pipeline as the book — which is exactly why the credits
sound like the same narrator rather than a bolted-on announcement.

The wording is French, like the rest of the narration this fork produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "CLOSING_TITLE",
    "OPENING_TITLE",
    "BookCredits",
    "SYNTHETIC_DISCLOSURE",
]

#: Chapter titles used for the two credit files, and their marker names in the
#: assembled M4B.
OPENING_TITLE = "Générique de début"
CLOSING_TITLE = "Générique de fin"

#: Said when no human narrator is named. Not a disclaimer bolted on for safety:
#: distributors require synthetic narration to be identified as such.
SYNTHETIC_DISCLOSURE = "une voix de synthèse"

# A title already carrying its author ("Autour de la Lune, par Jules Verne")
# would otherwise be announced as "…, par Jules Verne, de Jules Verne".
_AUTHOR_PREPOSITIONS = frozenset({"par", "de", "by"})


def _clean(text: str) -> str:
    """One line, single spaces, no trailing punctuation of its own."""
    collapsed = " ".join((text or "").split())
    return collapsed.rstrip(" .;,:")


def _sentence(text: str) -> str:
    """End a credit line on a full stop, so the narrator lands it."""
    text = _clean(text)
    return f"{text}." if text else ""


@dataclass(frozen=True)
class BookCredits:
    """What the recording says about itself, at its two ends."""

    title: str
    author: str = ""
    #: Human narrator. Left empty for a synthetic reading, which is then
    #: disclosed rather than passed off as a performance.
    narrator: str = ""
    subtitle: str = ""
    publisher: str = ""
    year: str = ""
    #: Public-domain works are worth saying so: it answers the rights question
    #: a distributor asks about every uploaded recording.
    public_domain: bool = False
    #: Turning this off is a deliberate act — see the module docstring.
    disclose_synthetic: bool = True

    @property
    def narrator_credit(self) -> str:
        """Who the recording says read it."""
        narrator = _clean(self.narrator)
        if narrator:
            return narrator
        return SYNTHETIC_DISCLOSURE if self.disclose_synthetic else ""

    def _work(self) -> str:
        """« Title », de Author — the phrase both credits are built around."""
        title = _clean(self.title) or "Ce livre"
        author = _clean(self.author)
        piece = f"« {title} »"
        if self.subtitle:
            piece += f", {_clean(self.subtitle)}"
        if author and not _names_the_author(title, author):
            piece += f", de {author}"
        return piece

    def opening(self) -> str:
        """The first thing heard: title, subtitle, author, narrator."""
        lines: List[str] = [_sentence(self._work())]
        narrator = self.narrator_credit
        if narrator:
            lines.append(_sentence(f"Lu par {narrator}"))
        return "\n\n".join(line for line in lines if line)

    def closing(self) -> str:
        """The last thing heard: the work named again, then the production."""
        narrator = self.narrator_credit
        first = f"Vous venez d'écouter {self._work()}"
        if narrator:
            first += f", lu par {narrator}"
        lines: List[str] = [_sentence(first)]

        publisher = _clean(self.publisher)
        year = _clean(self.year)
        if publisher and year:
            lines.append(_sentence(f"Enregistrement produit par {publisher}, {year}"))
        elif publisher:
            lines.append(_sentence(f"Enregistrement produit par {publisher}"))
        elif year:
            lines.append(_sentence(f"Enregistrement réalisé en {year}"))

        if self.public_domain:
            lines.append(_sentence("Texte du domaine public"))
        return "\n\n".join(line for line in lines if line)

    def missing_for_distribution(self) -> List[str]:
        """What a distributor would send this recording back for.

        Reported rather than raised: a draft narration is a perfectly reasonable
        thing to produce, and the gaps only matter on the day it is uploaded.
        """
        missing: List[str] = []
        if not _clean(self.title):
            missing.append("le titre")
        if not _clean(self.author):
            missing.append("l'auteur")
        if not self.narrator_credit:
            missing.append("le narrateur (ou la mention de voix de synthèse)")
        return missing


def _names_the_author(title: str, author: str) -> bool:
    """Whether the title already ends by naming the author.

    Matched from the end rather than by searching forwards: "Autour **de** la
    Lune, par Jules Verne" has a ``de`` long before the one that matters.
    """
    title_key = _clean(title).casefold()
    author_key = _clean(author).casefold()
    if not author_key or not title_key.endswith(author_key):
        return False
    head = title_key[: -len(author_key)].strip()
    if head.endswith((",", "-", "—", "–", ":")):
        return True
    words = head.rstrip(" ,-—–:").split()
    return bool(words) and words[-1] in _AUTHOR_PREPOSITIONS
