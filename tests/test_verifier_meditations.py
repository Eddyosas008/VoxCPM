"""Tests de scripts/verifier_meditations.py — ce qui coûte une séance ratée est nommé.

Calibrage : sur la série 01, l'estimation à cent trente mots par minute a
donné 2,1 / 6,0 / 9,2 minutes pour des MP3 de 2,2 / 6,0 / 9,3.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "verifier_meditations", ROOT / "scripts" / "verifier_meditations.py"
)
verifier_meditations = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# Enregistré avant l'exécution : le module définit une dataclass, et
# dataclasses va chercher le module dans sys.modules pour la construire.
sys.modules["verifier_meditations"] = verifier_meditations
spec.loader.exec_module(verifier_meditations)

MENTION = ("Cette séance accompagne votre bien-être. Elle ne remplace pas l'avis d'un "
           "professionnel de santé.")


def _script(mots: int, silences: str = "[PAUSE:DEEP] " * 30) -> str:
    corps = " ".join(["Respirez doucement et laissez le corps se poser."] * (mots // 8))
    return f"Bienvenue. {corps} {silences}\n\n{MENTION}\n"


def test_un_script_conforme_passe(tmp_path):
    f = tmp_path / "01_test.txt"
    f.write_text(_script(1350), encoding="utf-8")
    v = verifier_meditations.verifier(f, 15.0, 1.5)
    assert v.ok, v.defauts
    assert 1300 <= v.mots <= 1450 and v.silence_s == 300


def test_la_duree_hors_cible_est_nommee(tmp_path):
    f = tmp_path / "02_court.txt"
    f.write_text(_script(600), encoding="utf-8")
    v = verifier_meditations.verifier(f, 15.0, 1.5)
    assert any("durée estimée" in d for d in v.defauts)


def test_les_pieges_du_moteur_sont_nommes(tmp_path):
    f = tmp_path / "03_pieges.txt"
    f.write_text(
        "Inspirez 5 fois (lentement). Expirez tout. [RALENTIR] [TRUC] "
        "Voir la suite ; etc. — fin.\n", encoding="utf-8")
    v = verifier_meditations.verifier(f, 15.0, 100.0)
    texte = " ; ".join(v.defauts)
    for attendu in ("chiffre", "parenthèse", "Expirez", "repère inconnu",
                    "indication d'interprétation", "point-virgule", "mention finale", "abréviation"):
        assert attendu in texte, attendu


def test_un_silence_chiffre_nest_pas_un_chiffre_prononce(tmp_path):
    f = tmp_path / "04_silence.txt"
    f.write_text(_script(1350, "[SILENCE 20] " * 15), encoding="utf-8")
    v = verifier_meditations.verifier(f, 15.0, 1.5)
    assert not any("chiffre" in d for d in v.defauts)
    assert v.silence_s == 300


def test_le_markdown_est_refuse(tmp_path):
    f = tmp_path / "05_md.txt"
    f.write_text("# Titre\n" + _script(1350), encoding="utf-8")
    assert any("markdown" in d for d in verifier_meditations.verifier(f, 15.0, 1.5).defauts)
