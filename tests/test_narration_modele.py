"""Tests de narration.modele — charger la révision qu'on a, pas celle qu'on n'a pas.

Le cache Hugging Face est reconstitué dans un dossier temporaire : un snapshot
complet, un snapshot plus neuf mais sans poids (le téléchargement a lâché), et
on vérifie que la résolution choisit le complet, dit sa révision, et ne va sur
le hub que si on le lui demande ou s'il n'y a rien.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import modele  # noqa: E402

REPO = "openbmb/VoxCPM2"
COMPLET = "bffb3df5a29440629464e5e839f4d214c8714c3d"
INCOMPLET = "32279effe8c19989596f05d353d1447f51d9e915"


def _snapshot(cache: Path, revision: str, poids: bool, age: float = 0.0) -> Path:
    s = cache / "models--openbmb--VoxCPM2" / "snapshots" / revision
    s.mkdir(parents=True)
    (s / "config.json").write_text("{}", encoding="utf-8")
    (s / "tokenizer.json").write_text("{}", encoding="utf-8")
    if poids:
        p = s / "model.safetensors"
        p.write_bytes(b"\x00" * 16)
        if age:
            t = time.time() - age
            os.utime(p, (t, t))
    return s


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "hub"


def test_le_snapshot_complet_est_prefere_au_plus_neuf_sans_poids(cache):
    _snapshot(cache, COMPLET, poids=True)
    _snapshot(cache, INCOMPLET, poids=False)
    r = modele.resoudre(REPO, cache)
    assert r.origine == "cache" and r.revision == COMPLET
    assert r.chemin.endswith(COMPLET)
    assert "bffb3df" in r.describe() and "sans réseau" in r.describe()


def test_entre_deux_complets_le_plus_recent_lemporte(cache):
    _snapshot(cache, "a" * 40, poids=True, age=3600)
    _snapshot(cache, "b" * 40, poids=True)
    assert modele.resoudre(REPO, cache).revision == "b" * 40


def test_sans_rien_en_cache_on_laisse_le_hub_faire(cache):
    r = modele.resoudre(REPO, cache)
    assert r.origine == "hub" and r.chemin == REPO
    assert "hub" in r.describe()


def test_un_snapshot_sans_poids_ne_compte_pas(cache):
    _snapshot(cache, INCOMPLET, poids=False)
    assert modele.snapshots_complets(REPO, cache) == []
    assert modele.resoudre(REPO, cache).origine == "hub"


def test_en_ligne_ignore_le_cache_expres(cache):
    """C'est ainsi, et seulement ainsi, qu'on adopte une nouvelle révision."""
    _snapshot(cache, COMPLET, poids=True)
    r = modele.resoudre(REPO, cache, en_ligne=True)
    assert r.origine == "hub" and r.chemin == REPO


def test_un_chemin_explicite_reste_un_chemin(tmp_path, cache):
    local = tmp_path / "mon_modele"
    local.mkdir()
    r = modele.resoudre(str(local), cache)
    assert r.origine == "chemin" and r.chemin == str(local)
    assert "chemin local" in r.describe()


def test_le_cache_suit_les_variables_de_hugging_face(tmp_path):
    assert modele.cache_hub({"HF_HUB_CACHE": str(tmp_path / "x")}) == tmp_path / "x"
    assert modele.cache_hub({"HF_HOME": str(tmp_path / "h")}) == tmp_path / "h" / "hub"
    assert modele.cache_hub({}) == Path.home() / ".cache" / "huggingface" / "hub"


def test_le_module_nimporte_ni_torch_ni_huggingface_hub():
    import ast

    arbre = ast.parse(Path(modele.__file__).read_text(encoding="utf-8"))
    noms = set()
    for n in arbre.body:
        if isinstance(n, ast.Import):
            noms.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            noms.add(n.module.split(".")[0])
    assert not noms & {"torch", "huggingface_hub", "transformers"}


class TestDansNarrateBook:
    """Le pré-vol dit quelle révision sera chargée, et le moteur la reçoit."""

    def test_le_prevol_nomme_la_revision_du_cache(self, monkeypatch, tmp_path, capsys):
        from test_narrate_book_qc import narrate_book  # noqa: F401 - stub du moteur

        cache = tmp_path / "hub"
        _snapshot(cache, COMPLET, poids=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(cache))
        livre = tmp_path / "livre.txt"
        livre.write_text("Une phrase de test suffisante pour un segment.\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "narrate_book.py", str(livre), "--voice", "Voix de test",
            "--outdir", str(tmp_path / "out"), "--no-credits", "--dry-run",
        ])
        assert narrate_book.main() == 0
        sortie = capsys.readouterr().out
        assert "Modèle      : openbmb/VoxCPM2 — révision bffb3df en cache local" in sortie

    def test_le_moteur_recoit_le_chemin_resolu_et_le_cache_garde_lidentifiant(
        self, monkeypatch, tmp_path
    ):
        import test_narrate_book_qc as stub
        from narration import repair

        cache = tmp_path / "hub"
        snapshot = _snapshot(cache, COMPLET, poids=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(cache))
        recus = []
        original = stub.StubDemo.__init__

        def init(self, **kwargs):
            recus.append(kwargs.get("model_id"))
            original(self, **kwargs)

        monkeypatch.setattr(stub.StubDemo, "__init__", init)
        livre = tmp_path / "livre.txt"
        livre.write_text("Une phrase de test suffisante pour un segment.\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "narrate_book.py", str(livre), "--voice", "Voix de test",
            "--outdir", str(tmp_path / "out"), "--no-credits",
        ])
        assert stub.narrate_book.main() == 0
        assert recus == [str(snapshot)]
        # La clé du cache de segments ne bouge pas avec l'emplacement du cache.
        assert repair.BookPlan.load(tmp_path / "out").voice["model_id"] == "openbmb/VoxCPM2"
