"""Faire relire chaque segment par une machine, pour trouver ceux qui s'arrêtent en route.

Le contrôle qualité mesure l'audio contre le texte : durée, débit, silences,
boucles. Il attrape le segment muet, le segment qui babille, le segment coupé
net. Il n'attrape pas le segment **qui s'arrête poliment au milieu de sa
phrase** — une phrase coupée en deux a un débit normal, une fin normale, un
niveau normal. Mesuré sur un livre narré : les segments tronqués s'étalent de
0,84× à 2,82× le débit médian, les sains montent à 1,84×. Aucun seuil de
débit ne les sépare. Seule la relecture tranche : on fait transcrire l'audio
par une reconnaissance vocale, et on regarde **jusqu'où dans le texte** la
transcription va.

Le bon instrument n'est pas la couverture (quels mots de la source sont
retrouvés, dans n'importe quel ordre) mais la **portée alignée** : la position
du dernier mot de la source qu'un alignement ordonné retrouve dans ce qui a été
entendu. La couverture confond troncature et réécriture — Whisper écrit « 1980 »
quand le texte dit « mille neuf cent quatre-vingts » — et une simple
appartenance se fait piéger par un mot répété : un segment coupé à 16 % a rendu
92 % de « couverture » parce que son dernier mot entendu revenait en fin de
texte. L'alignement ordonné (plus longue sous-séquence commune) ne tombe pas
dans ce piège, et il tolère les paraphrases de la transcription au milieu.

Ce module est sans torch, comme le reste du paquet : la transcription arrive
sous forme d'un appelable, ce qui permet de tester la logique en millisecondes.
:func:`transcripteur_whisper` fabrique le vrai, et n'importe torch qu'à l'appel.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import cache as cache_tools
from . import repair

__all__ = [
    "MODELE",
    "Alignement",
    "Relecture",
    "Seuils",
    "aligner",
    "juger",
    "mots",
    "normaliser",
    "pliable",
    "recoller_sigles",
    "relire_livre",
    "resume",
    "transcripteur_whisper",
]

#: Le modèle de reconnaissance vocale. Le « turbo » lit le français aussi bien
#: que le grand modèle pour un quart du temps ; c'est celui de l'audit.
MODELE = "openai/whisper-large-v3-turbo"

#: Le nom du rapport écrit à côté des chapitres.
RAPPORT = "relecture_report.json"


# --------------------------------------------------------------------------
# Jetons comparables
# --------------------------------------------------------------------------


def mots(texte: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÿ'’-]+", texte or "")


def pliable(mot: str) -> str:
    """Forme comparable : sans accent, sans casse, sans trait d'union.

    Whisper ponctue et accentue à sa façon ; une différence d'accent n'est pas
    une différence de prononciation. Le « çe » du lexique redevient « ce », et
    « A C E » recollé redevient « ACE ».
    """
    plat = unicodedata.normalize("NFKD", mot.lower())
    plat = "".join(c for c in plat if not unicodedata.combining(c))
    return plat.replace("-", "").replace("'", "").replace("’", "")


def recoller_sigles(jetons: Sequence[str]) -> List[str]:
    """« T. D. A. H. » redevient « TDAH », et « A C E » redevient « ACE »."""
    sortie: List[str] = []
    tampon: List[str] = []
    for j in list(jetons) + [""]:
        if len(j) == 1 and j.isalpha():
            tampon.append(j)
            continue
        if len(tampon) >= 2:
            sortie.append("".join(tampon))
        else:
            sortie.extend(tampon)
        tampon = []
        if j:
            sortie.append(j)
    return sortie


#: Les nombres écrits en toutes lettres par le normaliseur reviennent en
#: chiffres de la transcription. Ils ne peuvent pas s'aligner, donc ils ne
#: comptent ni d'un côté ni de l'autre : sans ça, une phrase qui finit par
#: « en deux mille vingt » semblerait tronquée de trois mots.
_NOMBRES = set("""
zéro un une deux trois quatre cinq six sept huit neuf dix onze douze treize
quatorze quinze seize vingt vingts trente quarante cinquante soixante cent
cents mille milles million millions milliard milliards
""".split())
_NOMBRES_PLATS = {pliable(n) for n in _NOMBRES}


def normaliser(texte: str) -> List[str]:
    """Les jetons d'un texte, tels qu'on les compare."""
    sortie = []
    for jeton in recoller_sigles(mots(texte)):
        plat = pliable(jeton)
        if not plat or plat.isdigit() or plat in _NOMBRES_PLATS:
            continue
        sortie.append(plat)
    return sortie


# --------------------------------------------------------------------------
# Alignement
# --------------------------------------------------------------------------


def _semblables(a: str, b: str) -> bool:
    """Deux jetons qui disent le même mot.

    L'égalité d'abord. Puis une tolérance pour les noms propres que la
    reconnaissance orthographie autrement (« Filiozat » / « Filliozat ») :
    assez longs, même début, et presque identiques. Le test de début garde la
    comparaison coûteuse pour les rares paires qui la méritent.
    """
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5 or a[:2] != b[:2]:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.8


@dataclass(frozen=True)
class Alignement:
    """Ce qu'un alignement ordonné de la source sur l'entendu a retrouvé."""

    #: Nombre de jetons de la source.
    n_source: int
    #: Nombre de jetons de la source retrouvés, dans l'ordre.
    alignes: int
    #: Position (1-based) du dernier jeton de la source retrouvé ; 0 si aucun.
    dernier: int

    @property
    def portee(self) -> float:
        """Jusqu'où dans la source la transcription est allée, de 0 à 1."""
        return self.dernier / self.n_source if self.n_source else 1.0

    @property
    def couverture(self) -> float:
        return self.alignes / self.n_source if self.n_source else 1.0

    @property
    def manquants_en_fin(self) -> int:
        """Les jetons de la source après le dernier retrouvé."""
        return self.n_source - self.dernier


def aligner(source: Sequence[str], entendu: Sequence[str]) -> Alignement:
    """Plus longue sous-séquence commune, reconstruite en appariant au plus tôt.

    Apparier au plus tôt compte : quand le dernier mot entendu existe aussi plus
    loin dans la source, l'apparier là-bas gonflerait la portée d'un segment
    tronqué. À longueur d'alignement égale, on prend toujours la position la
    plus précoce dans la source.
    """
    n, m = len(source), len(entendu)
    if n == 0:
        return Alignement(0, 0, 0)
    if m == 0:
        return Alignement(n, 0, 0)

    # L[i][j] : longueur de la LCS de source[i:] et entendu[j:].
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        si = source[i]
        row, below = L[i], L[i + 1]
        for j in range(m - 1, -1, -1):
            if _semblables(si, entendu[j]):
                row[j] = below[j + 1] + 1
            else:
                row[j] = max(below[j], row[j + 1])

    alignes = 0
    dernier = 0
    i = j = 0
    while i < n and j < m:
        if _semblables(source[i], entendu[j]) and L[i][j] == L[i + 1][j + 1] + 1:
            alignes += 1
            dernier = i + 1
            i += 1
            j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return Alignement(n, alignes, dernier)


# --------------------------------------------------------------------------
# Jugement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Seuils:
    """Ce qui sépare « tronqué » de « transcrit autrement ».

    Les trois conditions se cumulent. La portée seule condamnerait un segment
    de huit mots dont Whisper a avalé le dernier ; le nombre de mots manquants
    seul condamnerait un long segment paraphrasé en fin ; et un segment trop
    court ne donne pas assez de mots pour juger.
    """

    #: En deçà de cette portée, le segment s'arrête avant la fin de son texte.
    portee_min: float = 0.85
    #: ... et il manque au moins autant de mots en fin de source.
    manquants_min: int = 3
    #: Il faut au moins autant de mots dans la source pour juger.
    mots_min: int = 6


@dataclass
class Relecture:
    """Le verdict de la relecture d'un segment."""

    label: str
    source: str
    entendu: str
    alignement: Alignement
    #: Le segment compte assez de mots pour que le verdict veuille dire quelque chose.
    fiable: bool
    #: Le segment s'arrête avant la fin de son texte.
    tronque: bool

    @property
    def portee(self) -> float:
        return self.alignement.portee

    def describe(self) -> str:
        a = self.alignement
        etat = "TRONQUÉ" if self.tronque else ("ok" if self.fiable else "trop court pour juger")
        return (f"{etat} — portée {a.portee:.0%}, {a.alignes}/{a.n_source} mot(s) retrouvé(s), "
                f"{a.manquants_en_fin} manquant(s) en fin")

    def to_dict(self) -> dict:
        a = self.alignement
        return {
            "segment": self.label,
            "tronque": self.tronque,
            "fiable": self.fiable,
            "portee": round(a.portee, 3),
            "couverture": round(a.couverture, 3),
            "mots_source": a.n_source,
            "manquants_en_fin": a.manquants_en_fin,
            "source": self.source,
            "entendu": self.entendu,
        }


def juger(label: str, source: str, entendu: str, seuils: Seuils = Seuils()) -> Relecture:
    """Comparer ce qui a été demandé à ce qui a été entendu."""
    alignement = aligner(normaliser(source), normaliser(entendu))
    fiable = alignement.n_source >= seuils.mots_min
    tronque = bool(
        fiable
        and alignement.manquants_en_fin >= seuils.manquants_min
        and alignement.portee < seuils.portee_min
    )
    return Relecture(label, source, entendu, alignement, fiable, tronque)


# --------------------------------------------------------------------------
# Un livre entier
# --------------------------------------------------------------------------

Transcripteur = Callable[[int, np.ndarray], str]


def relire_livre(
    plan: repair.BookPlan,
    cache: cache_tools.ChunkCache,
    transcrire: Transcripteur,
    seuils: Seuils = Seuils(),
    cibles: Optional[Iterable[str]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Relecture]:
    """Relire chaque segment en cache d'un livre, ou seulement ``cibles``.

    Lit le cache plutôt que les chapitres finis, comme la réparation : un
    défaut se répare au segment, donc il se trouve au segment.
    """
    spec = plan.voice_spec()
    voulus = set(cibles) if cibles is not None else None
    travail: List[Tuple[str, str, str]] = []
    for chapitre in plan.chapters:
        for position, segment in enumerate(chapitre.segments, 1):
            label = repair.segment_label(chapitre.index, position)
            if voulus is not None and label not in voulus:
                continue
            travail.append((label, segment.text, cache.key(segment.text, spec)))

    verdicts: List[Relecture] = []
    for numero, (label, texte, cle) in enumerate(travail, 1):
        entree = cache.get(cle)
        if entree is None:
            continue
        sample_rate, wav = entree
        entendu = transcrire(sample_rate, wav)
        verdicts.append(juger(label, texte, entendu, seuils))
        if on_progress is not None:
            on_progress(numero, len(travail))
    return verdicts


def resume(verdicts: Sequence[Relecture], repares: Sequence[str] = ()) -> dict:
    """Le rapport d'une relecture, prêt à écrire en JSON.

    ``repares`` nomme les segments qu'une réparation a remis d'aplomb depuis ;
    « restants » est ce qu'il reste à l'oreille.
    """
    tronques = [v.label for v in verdicts if v.tronque]
    repares_set = set(repares)
    return {
        "segments": len(verdicts),
        "juges": sum(1 for v in verdicts if v.fiable),
        "tronques": len(tronques),
        "repares": len([t for t in tronques if t in repares_set]),
        "restants": [t for t in tronques if t not in repares_set],
        "details": [v.to_dict() for v in verdicts if v.tronque],
    }


# --------------------------------------------------------------------------
# Le vrai transcripteur
# --------------------------------------------------------------------------


def _vers_16k(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sample_rate == 16000:
        return x
    n = max(1, int(len(x) * 16000 / sample_rate))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")


def transcripteur_whisper(device: str = "cuda", modele: str = MODELE,
                          langue: str = "fr") -> Transcripteur:
    """Charger Whisper une fois et rendre « (sample_rate, audio) -> texte ».

    N'importe torch et transformers qu'ici, pour que le module reste importable
    sans eux. Le ``pipeline()`` de transformers 5 décode l'audio via torchcodec,
    dont les DLL réclament un ffmpeg partagé : on donne l'audio déjà décodé.
    """
    import torch  # noqa: WPS433 - import différé, voulu
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(modele)
    reseau = WhisperForConditionalGeneration.from_pretrained(modele).to(device).eval()

    def transcrire(sample_rate: int, wav: np.ndarray) -> str:
        x = _vers_16k(wav, sample_rate)
        entrees = proc(x, sampling_rate=16000, return_tensors="pt").input_features
        # Whisper se charge en demi-précision sur GPU : lui donner du float32
        # lève « Input type (float) and bias type (c10::Half) should be the same ».
        entrees = entrees.to(device=device, dtype=reseau.dtype)
        with torch.no_grad():
            ids = reseau.generate(entrees, language=langue, task="transcribe",
                                  max_new_tokens=440)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcrire
