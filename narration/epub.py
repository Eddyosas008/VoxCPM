"""Read an .epub and hand the pipeline the same thing a .txt would.

An EPUB is a ZIP holding XHTML documents, an OPF manifest that lists them and a
spine that puts them in reading order. Everything downstream of this module —
segmentation, French normalisation, synthesis, assembly — already works on plain
chapters separated by ``---``, so the whole job here is to turn a book into that
text and then get out of the way. Nothing is written to disk and nothing is
extracted: entries are read from the archive by name, so a crafted path in a
manifest cannot escape anywhere.

Four things earn their complexity:

* **Reading order comes from the spine, not from the file names.** Sorting the
  XHTML files alphabetically puts chapter 10 before chapter 2 and scatters the
  front matter, which is exactly the kind of error you only notice six hours
  into a narration.
* **Chapter titles are looked up in the table of contents** (the EPUB 3 nav
  document, or the EPUB 2 NCX) before falling back to the first heading in the
  document. Many books style their headings with a plain ``<div>``, so the
  heading is not always there — but the TOC nearly always is, and its titles are
  what a listener expects to see as chapter markers.
* **A file is not a chapter.** Books converted from a single HTML source — every
  Project Gutenberg book — are cut into fixed-size files that begin and end
  mid-chapter. Left alone, *Autour de la Lune* is six 60,000-character blocks
  instead of its twenty-five chapters, so files holding several chapters are cut
  at their headings and a file's opening fragment rejoins the chapter it
  continues. See :func:`_cut_level` for how the chapter heading level is found.
* **DRM is detected and refused up front.** An encrypted EPUB parses fine and
  yields binary noise; failing early with a clear reason beats narrating that.

Covers and half-titles are usually a handful of characters and would each become
their own one-line chapter, so ``min_chars`` drops the documents too short to be
worth a chapter of their own.
"""
from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote
from xml.etree import ElementTree as ET

__all__ = [
    "DEFAULT_MIN_CHARS",
    "EpubBook",
    "EpubChapter",
    "EpubError",
    "is_epub",
    "load_book_text",
    "read_epub",
    "summarize",
    "to_book_text",
]

#: Below this many characters a spine document is front matter (cover, colophon,
#: half-title), not a chapter. Low enough to keep a genuinely short prologue.
DEFAULT_MIN_CHARS = 140

_CONTAINER_PATH = "META-INF/container.xml"
_ENCRYPTION_PATH = "META-INF/encryption.xml"

# Tags whose boundaries are paragraph boundaries in the narration sense. `br` is
# handled apart: it breaks a line without ending a paragraph.
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt",
        "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "td", "th", "tr", "ul",
    }
)
_SKIPPED_TAGS = frozenset({"head", "script", "style", "svg", "template"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# Zero-width marks and BOMs survive `\s`, so a collapse leaves them glued
# inside a word on its way to the engine. Non-breaking spaces need no case of
# their own: the collapse turns them into ordinary ones, which is what a
# narrator reads anyway.
_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# A separator line in the assembled text would be read as a chapter break, so
# any the book itself contains has to stop looking like one.
_CHAPTER_SEPARATOR_RE = re.compile(r"(?m)^\s*---\s*$")


class EpubError(ValueError):
    """The file is not an EPUB we can read, and the reason is worth showing."""


@dataclass(frozen=True)
class EpubChapter:
    """One spine document, as narratable text."""

    title: str
    text: str
    #: Path of the source document inside the archive, for diagnostics.
    href: str = ""
    #: Whether the title came from the book (TOC or heading) rather than from
    #: the file name. An invented title makes a fine marker but must not be
    #: read aloud at the top of the chapter.
    titled: bool = True

    @property
    def characters(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class EpubBook:
    """A book, in reading order."""

    title: str
    author: str
    chapters: List[EpubChapter]
    #: Documents dropped as front matter, kept so the caller can say so.
    skipped: List[str]

    @property
    def characters(self) -> int:
        return sum(chapter.characters for chapter in self.chapters)


class _TextExtractor(HTMLParser):
    """XHTML to plain text, keeping paragraph breaks and where the headings are.

    Written against ``html.parser`` rather than an XML parser on purpose: EPUB
    content is nominally XHTML but real books ship unclosed tags and stray
    entities, and a strict parse would reject a book over a typo in its
    copyright page.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0
        self._heading_parts: List[str] = []
        self._heading_level = 0
        self._heading_start = 0
        #: (level, title, index in ``_parts`` where the heading opens).
        self.headings: List[Tuple[int, str, int]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._parts.append("\n")
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n\n")
        if tag in _HEADING_TAGS and not self._heading_level:
            self._heading_level = int(tag[1])
            self._heading_parts = []
            self._heading_start = len(self._parts)

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag.lower() == "br" and not self._skip_depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _HEADING_TAGS and self._heading_level == int(tag[1]):
            title = _collapse(" ".join(self._heading_parts))
            if title:
                self.headings.append((self._heading_level, title, self._heading_start))
            self._heading_level = 0
        if tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        self._parts.append(data)
        if self._heading_level:
            self._heading_parts.append(data)

    @property
    def text(self) -> str:
        return _normalize_whitespace("".join(self._parts))

    @property
    def heading(self) -> str:
        return self.headings[0][1] if self.headings else ""

    def sections(self, level: int) -> List[Tuple[str, str]]:
        """Cut this document at headings of ``level``, or return nothing.

        Returning nothing means "this document is one chapter" — which is the
        case whenever it holds a single heading of the cut level and opens on
        it, the ordinary one-file-per-chapter layout.

        Text before the first cut becomes an untitled opening section rather
        than being dropped: it is either front matter or, in books cut into
        fixed-size files, the tail of the chapter running into this one.
        """
        cuts = [(title, start) for lv, title, start in self.headings if lv == level]
        if not cuts:
            return []
        preamble = _normalize_whitespace("".join(self._parts[: cuts[0][1]]))
        if len(cuts) == 1 and not preamble:
            return []

        sections: List[Tuple[str, str]] = []
        if preamble:
            sections.append(("", preamble))
        for position, (title, start) in enumerate(cuts):
            end = cuts[position + 1][1] if position + 1 < len(cuts) else len(self._parts)
            text = _normalize_whitespace("".join(self._parts[start:end]))
            if text:
                sections.append((title, text))
        return sections


class _LinkCollector(HTMLParser):
    """Every ``<a href>`` in a document, in order, with its visible text.

    Enough to read an EPUB 3 nav document: its table of contents is a nested
    list of links, and the nesting only carries depth, which chapter markers do
    not use.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self._href: Optional[str] = None
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, _collapse(" ".join(self._parts))))
            self._href = None
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)


def _collapse(text: str) -> str:
    """One line, single-spaced."""
    return " ".join((text or "").split())


def _normalize_whitespace(text: str) -> str:
    """Single-spaced lines, blank line between paragraphs, nothing else."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Zero-width spaces and BOMs survive `\s` and would end up glued inside
    # a word sent to the engine. Non-breaking spaces need no special case:
    # the collapse below turns them into ordinary ones, which is what a
    # narrator reads anyway.
    text = _INVISIBLE_RE.sub("", text)
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _local(tag: str) -> str:
    """Local name of a possibly namespaced XML tag."""
    return tag.rsplit("}", 1)[-1].lower()


def _iter_local(root: ET.Element, name: str):
    """Descendants whose local name matches, namespace whatever it may be.

    Case-insensitive on both sides: NCX spells its elements ``navPoint`` while
    OPF spells everything in lower case.
    """
    name = name.lower()
    for element in root.iter():
        if _local(element.tag) == name:
            yield element


def _attr(element: ET.Element, name: str) -> str:
    """Attribute by local name — ``epub:type`` and ``type`` read the same."""
    name = name.lower()
    for key, value in element.attrib.items():
        if _local(key) == name:
            return value
    return ""


def _resolve(base: str, href: str) -> str:
    """Archive path of ``href`` written relative to the document at ``base``."""
    href = unquote((href or "").split("#", 1)[0].strip())
    if not href:
        return ""
    directory = posixpath.dirname(base)
    joined = posixpath.join(directory, href) if directory else href
    return posixpath.normpath(joined).lstrip("/")


def is_epub(path) -> bool:
    """Whether this path looks like an EPUB, by extension."""
    return Path(path).suffix.lower() == ".epub"


def _read(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as error:
        raise EpubError(f"Missing from the archive: {name}") from error


def _parse_xml(data: bytes, what: str) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        raise EpubError(f"Malformed {what}: {error}") from error


def _opf_path(archive: zipfile.ZipFile) -> str:
    """Where the manifest lives, per ``META-INF/container.xml``."""
    root = _parse_xml(_read(archive, _CONTAINER_PATH), "container.xml")
    for rootfile in _iter_local(root, "rootfile"):
        full_path = _attr(rootfile, "full-path")
        if full_path:
            return unquote(full_path).lstrip("/")
    raise EpubError("container.xml names no OPF file")


def _metadata(opf: ET.Element) -> Tuple[str, str]:
    """Title and author from the Dublin Core metadata, blank when absent."""
    title = author = ""
    for element in _iter_local(opf, "title"):
        title = _collapse(element.text or "")
        if title:
            break
    for element in _iter_local(opf, "creator"):
        author = _collapse(element.text or "")
        if author:
            break
    return title, author


def _manifest(opf: ET.Element, opf_path: str) -> Dict[str, Dict[str, str]]:
    """Manifest items by id, with archive paths already resolved."""
    items: Dict[str, Dict[str, str]] = {}
    for item in _iter_local(opf, "item"):
        item_id = _attr(item, "id")
        href = _attr(item, "href")
        if not item_id or not href:
            continue
        items[item_id] = {
            "path": _resolve(opf_path, href),
            "media_type": _attr(item, "media-type").lower(),
            "properties": _attr(item, "properties").lower(),
        }
    return items


def _spine_ids(opf: ET.Element) -> List[str]:
    """Reading order: the idrefs of the spine, linear items only.

    ``linear="no"`` marks material reachable from the text but outside its flow
    — notes, ads, pop-up figures. Narrating it would interleave footnotes with
    the prose.
    """
    order: List[str] = []
    for spine in _iter_local(opf, "spine"):
        for itemref in _iter_local(spine, "itemref"):
            idref = _attr(itemref, "idref")
            if idref and _attr(itemref, "linear").lower() != "no":
                order.append(idref)
        break
    return order


def _toc_from_nav(archive: zipfile.ZipFile, nav_path: str) -> Dict[str, str]:
    """Chapter titles by document path, read from an EPUB 3 nav document."""
    try:
        data = archive.read(nav_path)
    except KeyError:
        return {}
    collector = _LinkCollector()
    try:
        collector.feed(data.decode("utf-8", errors="replace"))
    except Exception:  # a broken TOC costs titles, never the book
        return {}
    titles: Dict[str, str] = {}
    for href, label in collector.links:
        target = _resolve(nav_path, href)
        if target and label and target not in titles:
            titles[target] = label
    return titles


def _toc_from_ncx(archive: zipfile.ZipFile, ncx_path: str) -> Dict[str, str]:
    """Chapter titles by document path, read from an EPUB 2 NCX."""
    try:
        root = _parse_xml(archive.read(ncx_path), "NCX")
    except (KeyError, EpubError):
        return {}
    titles: Dict[str, str] = {}
    for nav_point in _iter_local(root, "navPoint"):
        label = ""
        for text_element in _iter_local(nav_point, "text"):
            label = _collapse(text_element.text or "")
            if label:
                break
        source = ""
        for content in _iter_local(nav_point, "content"):
            source = _attr(content, "src")
            if source:
                break
        target = _resolve(ncx_path, source)
        if target and label and target not in titles:
            titles[target] = label
    return titles


def _table_of_contents(
    archive: zipfile.ZipFile,
    opf: ET.Element,
    manifest: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, str], str]:
    """Merged TOC titles, and the path of the TOC document itself.

    The nav document is part of the spine in many EPUB 3 books; returning its
    path lets the caller drop it rather than narrate a list of chapter names.
    """
    titles: Dict[str, str] = {}
    nav_path = ""
    for item in manifest.values():
        if "nav" in item["properties"].split():
            nav_path = item["path"]
            titles.update(_toc_from_nav(archive, nav_path))
            break

    for spine in _iter_local(opf, "spine"):
        toc_id = _attr(spine, "toc")
        if toc_id and toc_id in manifest:
            for path, label in _toc_from_ncx(archive, manifest[toc_id]["path"]).items():
                titles.setdefault(path, label)
        break

    return titles, nav_path


def _document(archive: zipfile.ZipFile, path: str) -> Optional[_TextExtractor]:
    """Parse one spine document, or ``None`` if it cannot be read."""
    try:
        data = archive.read(path)
    except KeyError:
        return None
    extractor = _TextExtractor()
    try:
        extractor.feed(data.decode("utf-8", errors="replace"))
        extractor.close()
    except Exception:  # one unparsable document must not lose the other forty
        return None
    return extractor


def _cut_level(documents: Sequence[_TextExtractor]) -> int:
    """The heading level that marks chapters, decided across the whole book.

    Within one document the question is unanswerable: a lone ``<h1>`` above
    repeated ``<h3>`` is a file holding twenty chapters, and it is equally one
    chapter subdivided into scenes. What tells them apart is which level *opens*
    the documents — that is the level at which the book was cut into files.

    One file per chapter: every document opens on its ``<h1>``, so ``<h1>`` is
    the chapter level and the ``<h3>`` scene headings inside are left alone.
    A book packed into fixed-size files: most documents open straight on an
    ``<h3>`` chapter heading, with the ``<h1>`` of the title page appearing in
    one file only, so ``<h3>`` wins and the packed chapters come apart.

    Zero means nothing to split on. A book that is a single document keeps
    whatever level opens it, so a one-file book stays one chapter — genuinely
    ambiguous, and the same thing a ``.txt`` without separators does.
    """
    openers: Dict[int, int] = {}
    with_headings = 0
    for document in documents:
        if not document.headings:
            continue
        with_headings += 1
        level = document.headings[0][0]
        openers[level] = openers.get(level, 0) + 1
    if not openers:
        return 0
    for level in sorted(openers):
        if openers[level] * 2 >= with_headings:
            return level
    return min(openers)


def _merge_short_sections(
    sections: Sequence[Tuple[str, str]], floor: int
) -> List[Tuple[str, str]]:
    """Fold sections too short to stand alone into the one before them.

    A heading with two lines under it is a section break, not a chapter. Folding
    keeps its words — dropping them would silently lose text from the book.
    """
    merged: List[Tuple[str, str]] = []
    for title, text in sections:
        if merged and len(text) < floor:
            previous_title, previous_text = merged[-1]
            body = f"{title}\n\n{text}" if title else text
            merged[-1] = (previous_title, f"{previous_text}\n\n{body}")
        else:
            merged.append((title, text))
    return merged


def _text_after_title(text: str, title: str) -> Optional[str]:
    """What follows the title when ``text`` opens with it, ignoring whitespace.

    Markup routinely breaks a heading across lines — a chapter number above its
    name — so the words match while the layout does not. Comparing whitespace-
    insensitively finds the title anyway, and returning the remainder lets the
    caller re-lay it as a single line.
    """
    title_index = text_index = 0
    while title_index < len(title) and text_index < len(text):
        if title[title_index].isspace():
            title_index += 1
        elif text[text_index].isspace():
            text_index += 1
        elif title[title_index].casefold() != text[text_index].casefold():
            return None
        else:
            title_index += 1
            text_index += 1
    if title[title_index:].strip():
        return None
    return text[text_index:].lstrip()


def _fallback_title(path: str, index: int) -> str:
    """A readable name when neither the TOC nor a heading gives one."""
    stem = posixpath.basename(path).rsplit(".", 1)[0]
    stem = _collapse(stem.replace("_", " ").replace("-", " "))
    return stem or f"Chapitre {index}"


def read_epub(
    path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    split_on_headings: bool = True,
) -> EpubBook:
    """Read an EPUB into chapters, in reading order.

    ``split_on_headings`` cuts a spine document that holds several chapters at
    its headings; turn it off to keep one chapter per file exactly as the book
    packages them.

    Raises :class:`EpubError` when the file is not a readable EPUB — a wrong
    extension, a corrupt archive, DRM, or a manifest that lists no text.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise EpubError(f"No such file: {file_path}")

    try:
        archive = zipfile.ZipFile(file_path)
    except zipfile.BadZipFile as error:
        raise EpubError("Not a readable EPUB: the file is not a valid ZIP archive") from error

    with archive:
        if _ENCRYPTION_PATH in archive.namelist():
            raise EpubError(
                "This EPUB is protected by DRM; its text cannot be read. "
                "Export or convert it to .txt first."
            )

        opf_path = _opf_path(archive)
        opf = _parse_xml(_read(archive, opf_path), "OPF manifest")
        title, author = _metadata(opf)
        manifest = _manifest(opf, opf_path)
        toc_titles, nav_path = _table_of_contents(archive, opf, manifest)

        documents: List[Tuple[str, _TextExtractor]] = []
        for idref in _spine_ids(opf):
            item = manifest.get(idref)
            if item is None:
                continue
            document_path = item["path"]
            if document_path == nav_path:
                continue
            if item["media_type"] and "html" not in item["media_type"]:
                continue

            extractor = _document(archive, document_path)
            if extractor is not None and extractor.text:
                documents.append((document_path, extractor))

    # Front matter is dropped by length — but a book made entirely of short
    # documents is a short book, not an empty one, so the floor is only applied
    # while something survives it.
    floor = max(1, min_chars)
    kept = [entry for entry in documents if len(entry[1].text) >= floor]
    skipped = [path for path, doc in documents if len(doc.text) < floor] if kept else []
    if not kept:
        kept = documents

    level = _cut_level([doc for _, doc in kept]) if split_on_headings else 0

    chapters: List[EpubChapter] = []
    for document_path, extractor in kept:
        text, heading = extractor.text, extractor.heading
        sections = extractor.sections(level) if level else []
        split = bool(sections)
        pieces = (
            _merge_short_sections(sections, floor)
            if split
            # Without a split the document's own heading titles it; with one,
            # that heading belongs to the first section, not to what precedes it.
            else [(toc_titles.get(document_path) or heading, text)]
        )
        # A heading shallower than the cut level, standing before the first cut,
        # titles what precedes it — a title page above the chapters it opens.
        opening_title = (
            heading if split and extractor.headings[0][0] < level else ""
        )

        for position, (piece_title, piece_text) in enumerate(pieces):
            if position == 0 and not piece_title:
                piece_title = opening_title

            # An untitled opening section in a split document is the tail of the
            # chapter that was already running — books cut into fixed-size files
            # start mid-chapter — so it continues it instead of pretending to be
            # a chapter of its own. Only the very first document has nothing to
            # continue, and there the section really is front matter.
            if split and position == 0 and not piece_title and chapters:
                previous = chapters[-1]
                chapters[-1] = replace(
                    previous, text=f"{previous.text}\n\n{piece_text}".strip()
                )
                continue

            chapter_title = piece_title or toc_titles.get(document_path)
            chapters.append(
                EpubChapter(
                    title=chapter_title
                    or _fallback_title(document_path, len(chapters) + 1),
                    text=piece_text,
                    href=document_path,
                    titled=bool(chapter_title),
                )
            )

    if not chapters:
        raise EpubError(
            "No readable text found in this EPUB — it may be a scanned book "
            "(images only) or use a structure we cannot read."
        )

    return EpubBook(title=title, author=author, chapters=chapters, skipped=skipped)


def to_book_text(book: EpubBook) -> str:
    """The ``---``-separated text the rest of the pipeline already narrates.

    Each chapter starts with its title on exactly one line, because that first
    line is what becomes the chapter marker downstream — and because a narrator
    does announce the chapter. When the text already opens with the title it is
    re-laid onto that single line rather than repeated, so a heading markup
    broke in two ("XIV" above its name) still yields a whole marker. A title we
    invented from a file name is never read aloud.
    """
    blocks: List[str] = []
    for chapter in book.chapters:
        text = _CHAPTER_SEPARATOR_RE.sub("* * *", chapter.text).strip()
        title = _collapse(chapter.title) if chapter.titled else ""
        if title:
            remainder = _text_after_title(text, title)
            text = f"{title}\n\n{text if remainder is None else remainder}".strip()
        blocks.append(text)
    return "\n\n---\n\n".join(blocks)


def load_book_text(
    path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    split_on_headings: bool = True,
) -> Tuple[str, EpubBook]:
    """Read an EPUB straight to narratable text, keeping the book for its metadata."""
    book = read_epub(path, min_chars=min_chars, split_on_headings=split_on_headings)
    return to_book_text(book), book


def summarize(book: EpubBook) -> str:
    """One line per chapter, for a plan the user reads before six hours of CPU."""
    lines = [
        f"« {book.title} »" + (f" — {book.author}" if book.author else "")
        if book.title
        else (book.author or "Sans titre"),
        f"{len(book.chapters)} chapitre(s) · {book.characters} caractères",
    ]
    for index, chapter in enumerate(book.chapters, 1):
        lines.append(f"  {index:>3}. {chapter.title} ({chapter.characters} car.)")
    if book.skipped:
        lines.append(f"  ({len(book.skipped)} document(s) trop court(s) ignoré(s))")
    return "\n".join(lines)
