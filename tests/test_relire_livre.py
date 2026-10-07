"""Tests de bout en bout de scripts/relire_livre.py.

Le moteur de synthèse est le stub des tests de qualité, et la reconnaissance
vocale est remplacée par un faux transcripteur qui retrouve le texte d'un
audio dans le cache lui-même — ce que Whisper ferait à l'oreille. Ce qui est
testé est la promesse du script : un segment qui s'arrête en route est
retrouvé, régénéré, relu, et son chapitre reconstruit — sans rien
synthétiser d'autre ; et quand aucun essai ne va au bout, c'est dit et compté.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import relecture, repair  # noqa: E402

from test_narrate_book_qc import BASE_SEED, CALLS, narrate_book  # noqa: E402,F401
from test_narrate_book_qc import reset_stub  # noqa: E402,F401  (fixture)

spec = importlib.util.spec_from_file_location("relire_livre", ROOT / "scripts" / "relire_livre.py")
relire_livre = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(relire_livre)


@pytest.fixture
def book(tmp_path):
    """Deux chapitres de longueurs franchement différentes : le faux
    transcripteur retrouve un segment par son audio, et le stub produit le
    même bruit pour deux textes de même longueur."""
    path = tmp_path / "livre.txt"
    path.write_text(
        "La mémoire de travail retient quelques éléments à la fois.\n"
        "---\n"
        "Une courte marche après le repas améliore nettement la glycémie, l'humeur "
        "et l'attention de tout l'après-midi qui suit.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def narrated(monkeypatch, book, tmp_path):
    outdir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "narrate_book.py", str(book), "--voice", "Voix de test",
        "--outdir", str(outdir), "--no-credits",
    ])
    assert narrate_book.main() == 0
    CALLS.clear()
    return outdir


def faux_transcripteur(cache_dir: Path, tronquer: str | None, repare_a_partir_de: int | None):
    """Retrouver le texte d'un audio dans le cache ; mentir sur un segment.

    ``tronquer`` est le texte dont la transcription s'arrête à la moitié tant
    que la prise en cache a un numéro d'essai inférieur à
    ``repare_a_partir_de`` (None : toujours tronqué).
    """
    def transcrire(sr, wav):
        for sidecar in cache_dir.glob("*.json"):
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            data, _ = sf.read(str(sidecar.with_suffix(".wav")), dtype="float32")
            if len(data) != len(wav) or not np.allclose(data, wav, atol=2e-3):
                continue
            texte, essai = meta["text"], int(meta.get("attempt", 0))
            if texte == tronquer and (repare_a_partir_de is None or essai < repare_a_partir_de):
                mots = texte.split()
                return " ".join(mots[: len(mots) // 2])
            return texte
        raise AssertionError("audio inconnu du cache")

    return transcrire


def lancer(monkeypatch, outdir, transcrire, *extra) -> int:
    monkeypatch.setattr(relecture, "transcripteur_whisper", lambda *a, **k: transcrire)
    monkeypatch.setattr(sys, "argv", ["relire_livre.py", str(outdir), "--device", "cpu", *extra])
    return relire_livre.main()


def rapport(outdir: Path) -> dict:
    return json.loads((outdir / relecture.RAPPORT).read_text(encoding="utf-8"))


class TestSansDefaut:
    def test_un_livre_sain_sort_zero_sans_rien_synthetiser(self, monkeypatch, narrated):
        faux = faux_transcripteur(narrated / ".cache", None, None)
        assert lancer(monkeypatch, narrated, faux, "--repair") == 0
        r = rapport(narrated)
        assert r["segments"] == 2 and r["tronques"] == 0 and r["restants"] == []
        assert CALLS == []

    def test_le_plan_garde_le_chemin_de_reference(self, narrated):
        """Vide pour une voix décrite — mais présent : c'est lui qui permet de
        réparer une voix clonée dans la bonne voix."""
        plan = repair.BookPlan.load(narrated)
        assert plan.voice["reference_path"] == ""


class TestTronque:
    def _cible(self, outdir: Path) -> str:
        return repair.BookPlan.load(outdir).segment(2, 1).text

    def test_sans_repair_la_troncature_est_signalee_et_rien_ne_bouge(
        self, monkeypatch, narrated, capsys
    ):
        faux = faux_transcripteur(narrated / ".cache", self._cible(narrated), None)
        assert lancer(monkeypatch, narrated, faux) == 1
        assert rapport(narrated)["restants"] == ["ch002/seg001"]
        assert "TRONQUÉ" in capsys.readouterr().out
        assert CALLS == []

    def test_un_segment_tronque_est_regenere_relu_et_son_chapitre_reconstruit(
        self, monkeypatch, narrated, capsys
    ):
        chapitre = narrated / "chapitre_002.wav"
        avant = chapitre.stat().st_mtime_ns
        faux = faux_transcripteur(narrated / ".cache", self._cible(narrated), 1)

        assert lancer(monkeypatch, narrated, faux, "--repair") == 0

        r = rapport(narrated)
        assert r["tronques"] == 1 and r["repares"] == 1 and r["restants"] == []
        # Un seul segment régénéré, avec une graine dérivée — jamais la graine
        # de base, qui redonnerait la même prise.
        assert len(CALLS) == 1 and CALLS[0][1] != BASE_SEED
        assert chapitre.stat().st_mtime_ns >= avant
        sortie = capsys.readouterr().out
        assert "Chapitre 002 reconstruit" in sortie
        assert "1 réparé(s), 0 restant(s)" in sortie

    def test_quand_aucun_essai_ne_va_au_bout_cest_dit_et_compte(
        self, monkeypatch, narrated, capsys
    ):
        faux = faux_transcripteur(narrated / ".cache", self._cible(narrated), None)
        code = lancer(monkeypatch, narrated, faux, "--repair", "--retries", "3")
        assert code == 1
        r = rapport(narrated)
        assert r["repares"] == 0 and r["restants"] == ["ch002/seg001"]
        assert len(CALLS) == 3
        sortie = capsys.readouterr().out
        assert "aucun essai ne va au bout" in sortie
        assert "0 réparé(s), 1 restant(s) à l'oreille : ch002/seg001" in sortie

    def test_le_rapport_peut_aller_ailleurs(self, monkeypatch, narrated, tmp_path):
        faux = faux_transcripteur(narrated / ".cache", None, None)
        cible = tmp_path / "ailleurs" / "r.json"
        assert lancer(monkeypatch, narrated, faux, "--json", str(cible)) == 0
        assert json.loads(cible.read_text(encoding="utf-8"))["segments"] == 2


class TestRefus:
    def test_sans_cache_rien_ne_peut_etre_relu(self, monkeypatch, narrated, capsys):
        for f in (narrated / ".cache").glob("*"):
            f.unlink()
        assert lancer(monkeypatch, narrated, lambda sr, wav: "") == 2
        assert "aucun cache" in capsys.readouterr().err

    def test_sans_plan_non_plus(self, monkeypatch, tmp_path, capsys):
        assert lancer(monkeypatch, tmp_path, lambda sr, wav: "") == 2

    def test_un_plan_qui_ne_retrouve_rien_dans_le_cache_nest_pas_un_livre_sain(
        self, monkeypatch, narrated, capsys
    ):
        """Vu sur un vrai dossier : la voix du plan ne correspondait plus au
        cache, et le script annonçait « 0 relu, 0 tronqué » avec un code zéro —
        indiscernable d'un livre parfait."""
        plan_path = narrated / repair.PLAN_FILENAME
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["voice"]["seed"] = 1
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        assert lancer(monkeypatch, narrated, lambda sr, wav: "") == 2
        assert "ne correspondent pas" in capsys.readouterr().err
        assert rapport(narrated)["segments"] == 0

    def test_une_voix_clonee_sans_son_enregistrement_nest_pas_reparee(
        self, monkeypatch, narrated, capsys
    ):
        """Mieux vaut un segment tronqué qu'un segment dans une autre voix."""
        plan_path = narrated / repair.PLAN_FILENAME
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["voice"]["reference"] = "0123456789abcdef"
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        # Le cache est adressé par la voix : le même texte sous la voix
        # « clonée » n'y est pas. On recopie les entrées sous la nouvelle clé.
        from narration import cache as cache_tools
        plan = repair.BookPlan.load(narrated)
        cache = cache_tools.ChunkCache(narrated / ".cache")
        ancienne = cache_tools.VoiceSpec(**{k: v for k, v in payload["voice"].items()
                                            if k in cache_tools.VoiceSpec.__dataclass_fields__
                                            and k != "reference"})
        for chapitre in plan.chapters:
            for segment in chapitre.segments:
                sr, wav = cache.get(cache.key(segment.text, ancienne))
                cache.put(cache.key(segment.text, plan.voice_spec()), sr, wav, text=segment.text)

        faux = faux_transcripteur(narrated / ".cache", self_cible(plan), None)
        assert lancer(monkeypatch, narrated, faux, "--repair") == 1
        assert "Réparation impossible" in capsys.readouterr().err
        assert CALLS == []


def self_cible(plan: repair.BookPlan) -> str:
    return plan.segment(2, 1).text


def test_echantillon_regulier():
    labels = [f"ch001/seg{i:03d}" for i in range(1, 11)]
    assert relire_livre._echantillon(labels, 0) is None
    assert relire_livre._echantillon(labels, 20) is None
    assert relire_livre._echantillon(labels, 5) == labels[::2]
