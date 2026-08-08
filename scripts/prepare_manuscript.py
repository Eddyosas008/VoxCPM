"""Turn a Markdown manuscript into text a narrator can read aloud.

A manuscript is written to be *seen*. Its first pages carry an ISBN, a
copyright notice, a table of contents and a web address — all of which a
narrator would never read out, and all of which a TTS engine reads out
happily. Its body carries asterisks for emphasis and hashes for headings,
which are silent on a page and absurd in an ear.

Three traps, in order of how much damage they do:

1. **``---`` means two different things.** In the manuscript it is a
   horizontal rule, sprinkled through the front matter. To ``narrate_book.py``
   it is *the* chapter separator. Converting naively cuts the book at the
   copyright page. So every rule is removed, and ``---`` is re-emitted only at
   the boundaries this script decides.
2. **Front matter is not narration.** Everything before the first content
   heading goes, except the blocks worth keeping (a disclaimer, a dedication),
   which are named rather than guessed.
3. **Tables cannot be read.** A Markdown table spoken aloud is a stream of
   pipes. They are dropped, and counted.

Nothing is removed silently: ``--report`` prints every dropped block, because
text taken out of a book has to be reported back to whoever asked for it.

    python scripts/prepare_manuscript.py manuscrit_complet.md -o livre.txt --report
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import List

# A heading that opens something a narrator actually reads. Anything before the
# first of these is front matter: title page, copyright, ISBN, contents.
CONTENT_HEADING = re.compile(
    r"^\s*(introduction|chapitre|partie|prologue|pr[ée]face|avant[- ]propos"
    r"|conclusion|[ée]pilogue|annexe|postface)\b",
    re.IGNORECASE,
)

# Front-matter blocks worth keeping anyway. A disclaimer carries legal weight
# and a dedication is read in most audiobooks; a copyright page is neither.
KEEP_IN_FRONT = re.compile(r"^\s*(avertissement|d[ée]dicace|note de l['’]auteur)\b", re.IGNORECASE)

# Blocks to drop even when they sit in the body.
DROP_ALWAYS = re.compile(r"^\s*(table des mati[èe]res|sommaire|remerciements?|bibliographie|index)\b", re.IGNORECASE)

# A part divider carries no prose of its own — it must not become a chapter of
# two words, it belongs to the chapter that follows.
PART_HEADING = re.compile(r"^\s*partie\b", re.IGNORECASE)

#: Un en-tête qui ne dit que son rang : « Chapitre 4 », « Introduction ». Le
#: manuscrit met la vraie formule juste en dessous, en niveau 2 — et comme le
#: marqueur du lecteur audio est la première ligne du chapitre, le sommaire
#: n'affichait que « Chapitre 4 ».
BARE_HEADING = re.compile(
    r"^\s*(chapitre\s+[0-9IVXLC]+|introduction|conclusion|[ée]pilogue|prologue"
    r"|avant[- ]propos|pr[ée]face|annexes?)\s*$",
    re.IGNORECASE,
)


def join_subtitle(chapter: str) -> str:
    """« Chapitre 1 » + « Le grand malentendu » = un titre qui dit quelque chose.

    Utile deux fois : le sommaire devient lisible sur un téléphone, et
    l'annonce sonne juste, parce qu'un narrateur lit le titre entier plutôt que
    son numéro seul.
    """
    blocs = chapter.split("\n\n")
    if len(blocs) < 2 or not BARE_HEADING.match(blocs[0].strip()):
        return chapter
    suite = blocs[1].strip()
    # Un sous-titre est court et ne se termine pas ; un paragraphe fait les deux.
    if not suite or len(suite) > 90 or suite.endswith((".", "!", "?", "…")):
        return chapter
    return "\n\n".join([f"{blocs[0].strip()} — {suite}"] + blocs[2:])


@dataclass
class Block:
    """A heading and the prose under it, down to the next heading of any level."""

    level: int
    title: str
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


#: A bracketed direction asking the narrator to stop. It is not a word, it is a
#: silence — so it becomes one, rather than being read out as "PAUSE".
PAUSE_MARKER = re.compile(r"\[\s*pause\s*\]", re.IGNORECASE)

#: Stage directions in a transcribed testimony. They tell a reader what
#: happened in the room; spoken aloud they say that the narrator laughed.
STAGE_DIRECTION = re.compile(
    r"\[\s*(rire|rires|hésitation|h[ée]sitations|silence(?:\s+prolong[ée])?|soupir|soupirs"
    r"|pleurs(?:\s+contenus)?|larmes|blanc|sanglots?|se l[èe]ve[^\]]*|s'?arr[êe]te[^\]]*)\s*\]",
    re.IGNORECASE,
)


def unbracket(text: str) -> tuple[str, list[str]]:
    """Deal with the square brackets a manuscript carries, by what they mean.

    A corpus of twenty books held 993 of them in 24 forms, and they are not one
    thing. ``[PAUSE]``, 969 times over, is an instruction to stop talking.
    ``[rire]`` is a stage direction in a transcribed testimony. But
    ``[nom du département]`` and ``[ton mari / ta femme]`` are the sentence
    itself — a blank the reader fills — and deleting them leaves a hole where
    the meaning was.

    So: a pause becomes a paragraph break, a stage direction goes, and anything
    else keeps its words and loses only its brackets. Brackets are never
    spoken; what is inside them sometimes is.
    """
    removed: list[str] = []

    def note(kind: str, m: re.Match) -> str:
        removed.append(f"{kind} : {m.group(0)}")
        return ""

    text = PAUSE_MARKER.sub(lambda m: note("pause", m) or "\n\n", text)
    text = STAGE_DIRECTION.sub(lambda m: note("didascalie", m), text)

    def keep_inside(m: re.Match) -> str:
        inner = m.group(1).strip()
        removed.append(f"crochets retirés : {m.group(0)}")
        return inner

    text = re.sub(r"\[([^\]\n]{1,80})\]", keep_inside, text)
    return text, removed


def strip_inline(text: str) -> str:
    """Remove the marks that are silent on a page and spoken by an engine."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images: nothing to say
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # links: keep the words
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)        # inline code
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    text = re.sub(r"^\s{0,3}>\s?", "", text)                  # blockquote marker
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text)              # bullet
    text = re.sub(r"^\s{0,3}\d+[.)]\s+", "", text)            # numbered item
    return text.strip()


def parse(md: str) -> tuple[List[Block], List[str]]:
    """Split Markdown into heading-led blocks, dropping what cannot be spoken."""
    removed: List[str] = []
    blocks: List[Block] = [Block(0, "")]
    in_fence = False
    in_table = False

    for raw in md.splitlines():
        if raw.strip().startswith("```"):
            in_fence = not in_fence
            if in_fence:
                removed.append("bloc de code")
            continue
        if in_fence:
            continue

        # A table row, and the ---|--- rule under it, are unreadable aloud.
        if re.match(r"^\s*\|", raw) or re.match(r"^\s*\|?[\s:-]*\|[\s:|-]*$", raw) and "|" in raw:
            if not in_table:
                removed.append("tableau")
                in_table = True
            continue
        in_table = False

        heading = re.match(r"^(#{1,6})\s+(.*)$", raw)
        if heading:
            blocks.append(Block(len(heading.group(1)), strip_inline(heading.group(2))))
            continue

        # Horizontal rules only ever meant "new visual section"; the chapter
        # boundaries this script emits are decided from the headings instead.
        if re.match(r"^\s*([-*_])\1{2,}\s*$", raw):
            continue

        blocks[-1].lines.append(strip_inline(raw))

    return [b for b in blocks if b.title or b.text], removed


def to_chapters(blocks: List[Block]) -> tuple[List[str], List[str]]:
    """Group blocks into chapters, and say what was left out."""
    removed: List[str] = []

    start = next((i for i, b in enumerate(blocks) if CONTENT_HEADING.match(b.title)), None)
    if start is None:
        # No recognisable structure — narrate the whole thing as one chapter
        # rather than refuse. Better a long chapter than no book.
        body = "\n\n".join(b.text for b in blocks if b.text)
        return ([body] if body else []), removed

    kept_front = []
    for b in blocks[:start]:
        if KEEP_IN_FRONT.match(b.title):
            kept_front.append(b)
        elif b.title or b.text:
            removed.append(f"liminaire : {b.title or b.text[:40]}…")

    chapters: List[str] = []
    current: List[str] = []

    for b in kept_front:
        chapters.append("\n\n".join(x for x in (b.title, b.text) if x))

    for b in blocks[start:]:
        if DROP_ALWAYS.match(b.title):
            removed.append(f"section : {b.title}")
            continue
        # A new chapter opens on a level-1 heading — except a part divider,
        # which introduces the chapter after it rather than standing alone.
        if b.level == 1 and not PART_HEADING.match(b.title):
            if current:
                chapters.append("\n\n".join(current).strip())
            current = []
        piece = "\n\n".join(x for x in (b.title, b.text) if x)
        if piece:
            current.append(piece)

    if current:
        chapters.append("\n\n".join(current).strip())

    chapters = [join_subtitle(c) for c in chapters]
    return [c for c in chapters if c.strip()], removed


def main() -> int:
    # The report names chapters, so it carries whatever the book does — and a
    # Windows console defaults to cp1252, where an em dash or an arrow raises
    # rather than prints. Never let the summary kill a conversion that worked.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manuscript", help="fichier Markdown")
    ap.add_argument("-o", "--output", help="fichier .txt de sortie (défaut : à côté du manuscrit)")
    ap.add_argument("--report", action="store_true", help="détailler ce qui a été retiré")
    args = ap.parse_args()

    src = pathlib.Path(args.manuscript)
    if not src.is_file():
        print(f"introuvable : {src}", file=sys.stderr)
        return 1

    md = src.read_text(encoding="utf-8", errors="replace")
    # Avant tout découpage : un « [PAUSE] » devenu saut de paragraphe doit
    # pouvoir séparer deux paragraphes, ce que le parseur lira ensuite.
    md, removed_brackets = unbracket(md)
    blocks, removed_parse = parse(md)
    removed_parse = removed_parse + removed_brackets
    chapters, removed_struct = to_chapters(blocks)

    if not chapters:
        print("aucun texte narrable trouvé", file=sys.stderr)
        return 1

    out = pathlib.Path(args.output) if args.output else src.with_suffix(".narration.txt")
    body = "\n\n---\n\n".join(chapters)
    # The separator must be unambiguous: it is the one thing narrate_book.py
    # keys on, so no stray rule may survive anywhere else in the file.
    assert body.count("\n---\n") == len(chapters) - 1, "séparateur ambigu"
    out.write_text(body + "\n", encoding="utf-8")

    chars = sum(len(c) for c in chapters)
    print(f"{len(chapters)} chapitre(s) · {chars} caractères · ~{chars/15/60:.0f} min → {out.name}")
    for i, c in enumerate(chapters, 1):
        print(f"  {i:>3}. {c.splitlines()[0][:62]:<62} {len(c):>7} car.")

    counts: dict[str, int] = {}
    for r in removed_parse:
        # Grouper par nature : 969 lignes « pause : [PAUSE] » n'apprennent rien
        # de plus qu'une seule ligne disant 969.
        cle = r.split(" : ")[0] if " : " in r else r
        counts[cle] = counts.get(cle, 0) + 1
    if counts or removed_struct:
        print("\nRetiré :")
        for k, n in sorted(counts.items()):
            print(f"  {n:>3} × {k}")
        if args.report:
            for r in removed_struct:
                print(f"      {r}")
        else:
            print(f"  {len(removed_struct)} bloc(s) liminaire(s)/section(s) — --report pour le détail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
