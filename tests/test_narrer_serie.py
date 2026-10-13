"""Tests de scripts/narrer_serie.py : reprise, nouvel essai, copie, titres.

``subprocess.run`` est remplacé par un faux qui fabrique (ou non) le MP3 que
narrate_book aurait produit ; ce qui est testé est l'orchestration.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("narrer_serie", ROOT / "scripts" / "narrer_serie.py")
narrer_serie = importlib.util.module_from_spec(spec)
sys.modules["narrer_serie"] = narrer_serie
assert spec.loader is not None
spec.loader.exec_module(narrer_serie)


def _serie(tmp_path):
    d = tmp_path / "serie-x"
    d.mkdir()
    (d / "01_le_matin_anxieux.txt").write_text("Texte un.", encoding="utf-8")
    (d / "02_la_nuit.txt").write_text("Texte deux.", encoding="utf-8")
    (d / "LISEZ-MOI.txt").write_text("index", encoding="utf-8")
    (d / "THEMES.md").write_text(
        "01 Le matin anxieux — commencer la journée. Outil : respirer.\n", encoding="utf-8")
    return d


def _lancer(monkeypatch, tmp_path, d, echecs_par_slug=None, *extra, repare=()):
    echecs = dict(echecs_par_slug or {})
    appels = []

    def run(cmd, stdout=None, stderr=None, cwd=None, env=None, **_):
        if cmd[1].endswith("repair_segment.py"):
            slug = Path(cmd[2]).name
            appels.append((slug, "réparation " + cmd[cmd.index("--codes") + 1]))
            texte = "Chapitre 001 reconstruit" if slug in repare else "rien à régénérer"
            return types.SimpleNamespace(returncode=0, stdout=texte)
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        appels.append((outdir.name, cmd[cmd.index("--title") + 1]))
        if echecs.get(outdir.name, 0) > 0:
            echecs[outdir.name] -= 1
            return types.SimpleNamespace(returncode=-1073741819)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{outdir.name}_complet.mp3").write_bytes(b"ID3")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(narrer_serie.subprocess, "run", run)
    monkeypatch.setattr(narrer_serie.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["narrer_serie.py", str(d), "--device", "cpu",
                                      "--outroot", str(tmp_path / "out"), *extra])
    return narrer_serie.main(), appels


def _narrations(appels):
    return [a for a in appels if not a[1].startswith("réparation")]


def test_chaque_script_devient_un_mp3_recopie_a_cote(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    code, appels = _lancer(monkeypatch, tmp_path, d)
    assert code == 0
    assert [a[0] for a in _narrations(appels)] == ["01_le_matin_anxieux", "02_la_nuit"]
    assert (d / "01_le_matin_anxieux_complet.mp3").exists()
    assert (d / "02_la_nuit_complet.mp3").exists()


def test_le_titre_vient_de_themes_sinon_du_nom(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    _, appels = _lancer(monkeypatch, tmp_path, d)
    assert dict(_narrations(appels)) == {"01_le_matin_anxieux": "Le matin anxieux", "02_la_nuit": "La nuit"}


def test_une_seance_rendue_est_sautee(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    _lancer(monkeypatch, tmp_path, d)
    code, appels = _lancer(monkeypatch, tmp_path, d)
    assert code == 0 and appels == []


def test_un_plantage_au_chargement_est_retente(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    code, appels = _lancer(monkeypatch, tmp_path, d, {"01_le_matin_anxieux": 2})
    assert code == 0
    assert [a[0] for a in _narrations(appels)].count("01_le_matin_anxieux") == 3


def test_un_echec_persistant_nempeche_pas_la_suivante(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    code, appels = _lancer(monkeypatch, tmp_path, d, {"01_le_matin_anxieux": 9})
    assert code == 1
    assert (d / "02_la_nuit_complet.mp3").exists()
    etat = json.loads((tmp_path / "out" / "serie-x" / "state.json").read_text(encoding="utf-8"))
    assert etat["01_le_matin_anxieux"]["status"] == "failed"
    assert etat["02_la_nuit"]["status"] == "done"


def test_only_limite_la_passe(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    _, appels = _lancer(monkeypatch, tmp_path, d, None, "--only", "1")
    assert [a[0] for a in _narrations(appels)] == ["01_le_matin_anxieux"]


def test_les_fins_coupees_sont_reparees_puis_la_seance_reassemblee(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    code, appels = _lancer(monkeypatch, tmp_path, d, repare={"01_le_matin_anxieux"})
    assert code == 0
    assert appels[:3] == [("01_le_matin_anxieux", "Le matin anxieux"),
                          ("01_le_matin_anxieux", "réparation abrupt_end"),
                          ("01_le_matin_anxieux", "Le matin anxieux")]
    # Rien de réparé pour la seconde : pas de réassemblage.
    assert appels[3:] == [("02_la_nuit", "La nuit"), ("02_la_nuit", "réparation abrupt_end")]


def test_reparer_vide_coupe_la_reparation(monkeypatch, tmp_path):
    d = _serie(tmp_path)
    _, appels = _lancer(monkeypatch, tmp_path, d, None, "--reparer", "")
    assert not any(a[1].startswith("réparation") for a in appels)
