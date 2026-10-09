"""Trouver le modèle là où il est déjà, et ne pas en changer sans le dire.

Trois narrations ont échoué le même soir avant la première seconde de
synthèse : le dépôt ``openbmb/VoxCPM2`` avait changé de révision sur le hub,
la bibliothèque a voulu télécharger les nouveaux poids — 4,6 Go — et le réseau
a lâché en route. La révision précédente, complète, dormait dans le cache à
côté. Deux défauts dans un : la narration dépendait du réseau pour un modèle
qu'elle avait déjà, et elle aurait adopté une **autre révision** du modèle sans
un mot — or les voix décrites sont définies par (description, graine) *pour un
modèle donné* : changer de révision, c'est changer les quatorze voix du
catalogue, et la clé du cache de segments n'en saurait rien.

D'où cette règle : un identifiant de dépôt se résout d'abord vers la révision
**complète** présente dans le cache local, et on ne va sur le hub que si on le
demande (``--model-online``) ou si le cache n'a rien. Un chemin explicite reste
un chemin explicite.

Stdlib seulement : la disposition du cache de Hugging Face est connue
(``models--org--nom/snapshots/<révision>/``), inutile d'importer la
bibliothèque pour la lire.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__all__ = ["Resolution", "cache_hub", "resoudre", "snapshots_complets"]

#: Ce qu'un snapshot doit contenir pour qu'on puisse charger le modèle sans
#: rien télécharger. Les poids d'abord — c'est eux qui manquaient.
_REQUIS = ("config.json",)
_POIDS = ("*.safetensors", "*.bin", "*.pth")


def cache_hub(env: Optional[dict] = None) -> Path:
    """Le dossier ``hub`` du cache Hugging Face, selon les mêmes variables."""
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class Resolution:
    """Où charger le modèle, et pourquoi là."""

    chemin: str
    #: ``"chemin"`` (donné tel quel), ``"cache"`` (révision complète trouvée),
    #: ``"hub"`` (laissé à la bibliothèque, réseau possible).
    origine: str
    revision: Optional[str] = None

    def describe(self) -> str:
        if self.origine == "chemin":
            return f"{self.chemin} (chemin local)"
        if self.origine == "cache":
            return f"révision {self.revision[:7]} en cache local, sans réseau"
        return f"{self.chemin} — résolu sur le hub (réseau, peut changer de révision)"


def _complet(snapshot: Path) -> bool:
    if not all((snapshot / f).is_file() for f in _REQUIS):
        return False
    return any(any(snapshot.glob(motif)) for motif in _POIDS)


def snapshots_complets(repo_id: str, cache: Optional[Path] = None) -> List[Path]:
    """Les snapshots de ``repo_id`` qu'on peut charger sans réseau, le plus
    récent d'abord — récent au sens du fichier de poids, pas du dossier."""
    cache = cache or cache_hub()
    dossier = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not dossier.is_dir():
        return []
    complets = [s for s in dossier.iterdir() if s.is_dir() and _complet(s)]

    def recence(s: Path) -> float:
        poids = [p for motif in _POIDS for p in s.glob(motif)]
        return max((p.stat().st_mtime for p in poids), default=0.0)

    return sorted(complets, key=recence, reverse=True)


def resoudre(model_id: str, cache: Optional[Path] = None, en_ligne: bool = False) -> Resolution:
    """Résoudre ce que l'utilisateur a nommé vers ce qu'on charge vraiment."""
    if Path(model_id).is_dir():
        return Resolution(str(model_id), "chemin")
    if en_ligne:
        return Resolution(model_id, "hub")
    trouves = snapshots_complets(model_id, cache)
    if trouves:
        return Resolution(str(trouves[0]), "cache", trouves[0].name)
    return Resolution(model_id, "hub")
