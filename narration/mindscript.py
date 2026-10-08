"""Aller chercher les scripts dans MindScript Audio, et y rapporter l'audio.

MindScript Audio (https://mindscriptaudio.com) écrit les scripts — méditations,
hypnoses, relaxations — et les garde dans sa base. Leur texte est brut, avec
les repères de silence entre crochets que :mod:`narration.cues` comprend. Ce
module est le pont : lister les scripts d'un compte, en prendre un, le mettre
sous la forme que ``narrate_book.py`` attend, et déposer le MP3 produit dans la
bibliothèque audio du compte.

Deux façons d'entrer. Une **clé d'API** (en-tête ``x-api-key``, créée dans les
réglages de l'app) ouvre ``/api/v1`` — mais l'app la réserve aux comptes Team
ou Enterprise, et ``/api/v1`` ne sait pas recevoir d'audio. Une **session**
(courriel + mot de passe, comme l'app elle-même) ouvre tout, dépôt compris.
Le client fait les deux, et dit lequel il emploie.

Stdlib seulement : urllib et un pot à cookies. La couche réseau arrive par un
``opener`` remplaçable, ce qui permet de tester chaque appel sans serveur.
"""
from __future__ import annotations

import hashlib
import http.cookiejar
import json
import mimetypes
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "URL_PAR_DEFAUT",
    "VOIX_PAR_LANGUE",
    "Client",
    "ErreurMindScript",
    "Script",
    "encoder_multipart",
    "selectionner",
    "slugifier",
]

URL_PAR_DEFAUT = "https://mindscriptaudio.com"

#: La voix qu'on donne à un script quand personne n'en nomme une. Le
#: catalogue n'a que des voix françaises : un script anglais exige ``--voice``.
VOIX_PAR_LANGUE: Dict[str, str] = {"fr": "Aurore — méditation guidée"}


class ErreurMindScript(RuntimeError):
    """L'app a répondu autre chose que ce qu'on lui demandait."""


# --------------------------------------------------------------------------
# Un script
# --------------------------------------------------------------------------


def slugifier(texte: str, longueur: int = 60) -> str:
    plat = unicodedata.normalize("NFKD", texte or "")
    plat = "".join(c for c in plat if not unicodedata.combining(c)).lower()
    plat = re.sub(r"[^a-z0-9]+", "-", plat).strip("-")
    return plat[:longueur].rstrip("-") or "script"


@dataclass
class Script:
    """Un script tel que l'app le rend, réduit à ce que la narration consomme."""

    id: str
    title: str
    content: str
    language: str = "fr"
    estimated_minutes: int = 0
    word_count: int = 0
    updated_at: str = ""
    project_id: str = ""
    config: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, d: dict) -> "Script":
        config = d.get("config") or d.get("configJson") or d.get("config_json") or {}
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except ValueError:
                config = {}
        return cls(
            id=str(d.get("id", "")),
            title=str(d.get("title") or "Sans titre"),
            content=str(d.get("content") or ""),
            language=str(d.get("language") or config.get("language") or "fr").lower()[:2],
            estimated_minutes=int(d.get("estimatedMinutes") or d.get("estimated_minutes") or 0),
            word_count=int(d.get("wordCount") or d.get("word_count") or 0),
            updated_at=str(d.get("updatedAt") or d.get("updated_at") or d.get("createdAt") or ""),
            project_id=str(d.get("projectId") or d.get("project_id") or ""),
            config=config if isinstance(config, dict) else {},
        )

    def slug(self) -> str:
        """Un nom de dossier lisible et unique : le titre, puis un bout d'id."""
        return f"{slugifier(self.title, 48)}-{self.id[:8] or 'x'}"

    def texte(self) -> str:
        """Le contenu, prêt pour ``narrate_book.py``.

        L'app rend du texte brut avec ses repères ; on ne fait que ranger :
        fins de ligne Unix, espaces de fin retirées, jamais plus d'une ligne
        vide d'affilée. Le titre n'est pas prononcé — une méditation commence
        dans le calme, et l'app ne le lit pas non plus.
        """
        lignes = [ligne.rstrip() for ligne in self.content.replace("\r\n", "\n").split("\n")]
        texte = "\n".join(lignes).strip()
        return re.sub(r"\n{3,}", "\n\n", texte) + "\n"

    def empreinte(self) -> str:
        """Ce qui change quand le script change : son texte, rien d'autre."""
        return hashlib.sha256(self.texte().encode("utf-8")).hexdigest()[:16]

    def voix_par_defaut(self) -> Optional[str]:
        return VOIX_PAR_LANGUE.get(self.language)


def selectionner(
    scripts: Iterable[Script],
    ids: Optional[Sequence[str]] = None,
    langue: Optional[str] = None,
    depuis: Optional[str] = None,
) -> List[Script]:
    """Les scripts voulus : par id (préfixe accepté), langue, date de mise à jour."""
    voulus = list(scripts)
    if ids:
        voulus = [s for s in voulus if any(s.id == i or s.id.startswith(i) for i in ids)]
    if langue:
        voulus = [s for s in voulus if s.language == langue.lower()[:2]]
    if depuis:
        voulus = [s for s in voulus if s.updated_at[:10] >= depuis[:10]]
    return voulus


# --------------------------------------------------------------------------
# Multipart, parce que urllib ne le fait pas
# --------------------------------------------------------------------------


def encoder_multipart(
    champs: Dict[str, str], fichier: Tuple[str, str, bytes, str]
) -> Tuple[bytes, str]:
    """``(corps, content-type)`` d'un envoi de formulaire avec un fichier.

    ``fichier`` est ``(nom du champ, nom du fichier, octets, type MIME)``.
    """
    frontiere = "----voxcpm" + uuid.uuid4().hex
    morceaux: List[bytes] = []
    for nom, valeur in champs.items():
        morceaux.append(
            (f"--{frontiere}\r\nContent-Disposition: form-data; name=\"{nom}\"\r\n\r\n"
             f"{valeur}\r\n").encode("utf-8")
        )
    champ, nom_fichier, octets, mime = fichier
    morceaux.append(
        (f"--{frontiere}\r\nContent-Disposition: form-data; name=\"{champ}\"; "
         f"filename=\"{nom_fichier}\"\r\nContent-Type: {mime}\r\n\r\n").encode("utf-8")
    )
    morceaux.append(octets)
    morceaux.append(f"\r\n--{frontiere}--\r\n".encode("utf-8"))
    return b"".join(morceaux), f"multipart/form-data; boundary={frontiere}"


# --------------------------------------------------------------------------
# Le client
# --------------------------------------------------------------------------


class Client:
    """Parle à une instance de MindScript Audio.

    ``opener`` est ce qui envoie réellement les requêtes ; par défaut un
    ouvreur urllib qui garde les cookies de session. Les tests en donnent un
    faux et lisent ce qui lui a été demandé.
    """

    def __init__(
        self,
        base_url: str = URL_PAR_DEFAUT,
        api_key: Optional[str] = None,
        opener=None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.cookies = http.cookiejar.CookieJar()
        self.opener = opener or urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )
        self.timeout = timeout
        self.utilisateur: Optional[dict] = None

    # -- entrée ------------------------------------------------------------

    @property
    def mode(self) -> str:
        if self.utilisateur is not None:
            return "session"
        return "clé d'API" if self.api_key else "aucun"

    def connecter(self, email: str, mot_de_passe: str) -> dict:
        """Ouvrir une session comme l'app : POST /api/login, cookie gardé."""
        self.utilisateur = self._json("POST", "/api/login",
                                      corps={"email": email, "password": mot_de_passe})
        return self.utilisateur

    # -- scripts -----------------------------------------------------------

    def scripts(self) -> List[Script]:
        """Tous les scripts du compte, les plus récents d'abord."""
        if self.utilisateur is None and self.api_key:
            charge = self._json("GET", "/api/v1/scripts")
        else:
            charge = self._json("GET", "/api/scripts")
        liste = _deballer(charge)
        scripts = [Script.from_payload(d) for d in liste if isinstance(d, dict)]
        scripts.sort(key=lambda s: s.updated_at, reverse=True)
        return scripts

    def script(self, identifiant: str) -> Script:
        if self.utilisateur is None and self.api_key:
            charge = self._json("GET", f"/api/v1/scripts/{urllib.parse.quote(identifiant)}")
        else:
            charge = self._json("GET", f"/api/scripts/{urllib.parse.quote(identifiant)}")
        d = _deballer(charge)
        if not isinstance(d, dict):
            raise ErreurMindScript(f"réponse inattendue pour le script {identifiant}")
        return Script.from_payload(d)

    # -- audio -------------------------------------------------------------

    def televerser_audio(
        self,
        chemin: str | Path,
        nom: str,
        categorie: str = "custom",
        tags: Sequence[str] = ("mindscript", "voxcpm"),
    ) -> dict:
        """Déposer un fichier audio dans la bibliothèque du compte.

        Passe par ``POST /api/audio/assets/upload``, qui n'existe que sur la
        session : une clé d'API ne peut pas déposer, et c'est dit avant
        d'envoyer quoi que ce soit.
        """
        if self.utilisateur is None:
            raise ErreurMindScript(
                "le dépôt d'audio demande une session (courriel + mot de passe) ; "
                "l'API par clé ne sait que lire."
            )
        chemin = Path(chemin)
        mime = mimetypes.guess_type(chemin.name)[0] or "application/octet-stream"
        corps, content_type = encoder_multipart(
            {"name": nom, "category": categorie, "tags": ",".join(tags)},
            ("audioFile", chemin.name, chemin.read_bytes(), mime),
        )
        charge = self._requete("POST", "/api/audio/assets/upload", corps,
                               {"Content-Type": content_type})
        d = _deballer(json.loads(charge or b"{}"))
        return d if isinstance(d, dict) else {}

    # -- transport ---------------------------------------------------------

    def _json(self, methode: str, chemin: str, corps: Optional[dict] = None):
        octets = json.dumps(corps).encode("utf-8") if corps is not None else None
        entetes = {"Content-Type": "application/json"} if octets is not None else {}
        reponse = self._requete(methode, chemin, octets, entetes)
        if not reponse:
            return None
        try:
            return json.loads(reponse)
        except ValueError as error:
            raise ErreurMindScript(f"{methode} {chemin} : réponse qui n'est pas du JSON") from error

    def _requete(self, methode: str, chemin: str, corps: Optional[bytes],
                 entetes: Dict[str, str]) -> bytes:
        requete = urllib.request.Request(self.base_url + chemin, data=corps, method=methode)
        requete.add_header("Accept", "application/json")
        if self.api_key:
            requete.add_header("x-api-key", self.api_key)
        for cle, valeur in entetes.items():
            requete.add_header(cle, valeur)
        try:
            with self.opener.open(requete, timeout=self.timeout) as reponse:
                return reponse.read()
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = json.loads(error.read()).get("message") or ""
            except Exception:  # noqa: BLE001 - le détail est un bonus
                pass
            raise ErreurMindScript(
                f"{methode} {chemin} → {error.code}" + (f" : {detail}" if detail else "")
            ) from error
        except urllib.error.URLError as error:
            raise ErreurMindScript(f"{methode} {chemin} : {error.reason}") from error


def _deballer(charge):
    """``{"data": …}`` (API v1) ou la chose elle-même (API de session)."""
    if isinstance(charge, dict) and "data" in charge and len(charge) <= 3:
        return charge["data"]
    return charge
