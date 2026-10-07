"""La file refait le M4B et l'export ACX quand un segment a été retouché après coup.

Le défaut : ``narrate_book`` assemble et exporte à la fin de la narration ;
``repair_segment`` et la relecture passent *ensuite* et reconstruisent des
chapitres que plus personne n'assemble. Vingt-deux livres ont été livrés
ainsi, leurs corrections restées dans des WAV. La file doit donc relancer la
commande de narration — la même, celle qui a volé le plan — quand quelque
chose a changé, et seulement alors : elle coûte un chargement de modèle.

Les étapes sont des sous-processus ; on remplace ``run`` par un faux qui
tient le journal des commandes et fabrique les fichiers que chacune
produirait. Ce qui est testé est l'ordre, et la décision de relancer.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("narrate_queue", ROOT / "scripts" / "narrate_queue.py")
narrate_queue = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(narrate_queue)


@pytest.fixture
def file_dun_livre(tmp_path):
    qdir = tmp_path / "queue"
    (qdir / "textes").mkdir(parents=True)
    (qdir / "textes" / "livre-x.txt").write_text("Une phrase.\n", encoding="utf-8")
    qpath = qdir / "queue.json"
    qpath.write_text(json.dumps([{
        "slug": "livre-x", "txt": "textes/livre-x.txt", "voice": "Voix de test",
        "chars": 1000, "title": "X", "author": "Y",
    }]), encoding="utf-8")
    return qpath


def faux_run(scenario: dict, journal: list):
    """Chaque étape produit ce que la vraie produirait, selon le scénario."""

    def run(cmd, logfile=None):
        texte = " ".join(str(c) for c in cmd)
        journal.append(texte)
        if "narrate_book.py" in texte:
            if "--dry-run" in texte:
                return 0, ""
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "chapitre_001.wav").write_bytes(b"RIFF")
            (outdir / f"{outdir.name}_complet.m4b").write_bytes(b"\x00" * 10)
            (outdir / "acx").mkdir(exist_ok=True)
            (outdir / "acx" / "001 - X.mp3").write_bytes(b"\x00" * 10)
            return scenario.get("narration_rc", 0), ""
        if "repair_segment.py" in texte:
            if "--list" in texte:
                return 0, scenario.get("liste", "  (aucun défaut détecté)")
            return 0, scenario.get("reparation", "")
        if "relire_livre.py" in texte:
            outdir = Path(cmd[2])
            rapport = scenario.get("relecture")
            if rapport is None:
                return 2, "whisper absent"
            (outdir / "relecture_report.json").write_text(json.dumps(rapport), encoding="utf-8")
            return 1 if rapport.get("restants") else 0, ""
        return 0, ""

    return run


def lancer(monkeypatch, qpath, scenario, *extra):
    journal: list = []
    monkeypatch.setattr(narrate_queue, "run", faux_run(scenario, journal))
    monkeypatch.setattr(sys, "argv", [
        "narrate_queue.py", str(qpath), "--device", "cpu",
        "--outroot", str(qpath.parent.parent / "out"), "--sans-couverture", *extra,
    ])
    code = narrate_queue.main()
    etat = json.loads((qpath.parent / "state.json").read_text(encoding="utf-8"))
    return code, journal, etat["livre-x"]


def narrations(journal):
    return [c for c in journal if "narrate_book.py" in c and "--dry-run" not in c]


def position(journal, motif):
    return next(i for i, c in enumerate(journal) if motif in c)


RELECTURE_SAINE = {"segments": 40, "tronques": 0, "repares": 0, "restants": []}


class TestOrdreDesEtapes:
    def test_rien_retouche_une_seule_narration(self, monkeypatch, file_dun_livre):
        code, journal, etat = lancer(monkeypatch, file_dun_livre, {"relecture": RELECTURE_SAINE})
        assert code == 0 and etat["status"] == "done"
        assert len(narrations(journal)) == 1
        assert etat["truncated_found"] == 0 and etat["repaired"] == 0

    def test_la_relecture_passe_apres_la_reparation_et_avant_le_bilan(
        self, monkeypatch, file_dun_livre
    ):
        _, journal, _ = lancer(monkeypatch, file_dun_livre, {"relecture": RELECTURE_SAINE})
        assert position(journal, "repair_segment.py") < position(journal, "relire_livre.py")
        assert "--repair" in journal[position(journal, "relire_livre.py")]

    def test_un_segment_repare_par_la_relecture_refait_le_m4b(
        self, monkeypatch, file_dun_livre
    ):
        rapport = {"segments": 40, "tronques": 1, "repares": 1, "restants": []}
        code, journal, etat = lancer(monkeypatch, file_dun_livre, {"relecture": rapport})
        assert code == 0 and etat["status"] == "done"
        assert len(narrations(journal)) == 2
        # La seconde narration — l'assemblage — vient APRÈS la relecture, et
        # c'est la même commande : c'est elle qui a volé le plan.
        premiere, seconde = narrations(journal)
        assert premiere == seconde
        derniere = len(journal) - 1 - journal[::-1].index(seconde)
        assert derniere > position(journal, "relire_livre.py")
        assert etat["truncated_found"] == 1 and etat["truncated_repaired"] == 1

    def test_un_segment_repare_par_le_controle_qualite_refait_le_m4b_aussi(
        self, monkeypatch, file_dun_livre
    ):
        scenario = {"liste": "  ch001/seg003  FATAL  truncated\n",
                    "reparation": "  après : ok\n", "relecture": RELECTURE_SAINE}
        _, journal, etat = lancer(monkeypatch, file_dun_livre, scenario)
        assert len(narrations(journal)) == 2
        assert etat["repaired"] == 1

    def test_une_reparation_qui_a_garde_lancienne_prise_ne_refait_rien(
        self, monkeypatch, file_dun_livre
    ):
        scenario = {"liste": "  ch001/seg003  FATAL  truncated\n",
                    "reparation": "  le nouvel essai est moins bon, l'ancien est conservé\n",
                    "relecture": RELECTURE_SAINE}
        _, journal, etat = lancer(monkeypatch, file_dun_livre, scenario)
        assert len(narrations(journal)) == 1
        assert etat["repaired"] == 0

    def test_des_segments_restants_ne_bloquent_pas_le_livre(
        self, monkeypatch, file_dun_livre
    ):
        """Un segment que trois essais n'ont pas remis d'aplomb est dit, pas
        bloquant : le livre vaut mieux livré avec une phrase courte que pas livré."""
        rapport = {"segments": 40, "tronques": 2, "repares": 1, "restants": ["ch004/seg002"]}
        code, journal, etat = lancer(monkeypatch, file_dun_livre, {"relecture": rapport})
        assert code == 0 and etat["status"] == "done"
        assert len(narrations(journal)) == 2
        assert etat["truncated_found"] == 2 and etat["truncated_repaired"] == 1


class TestQuandLaRelectureNePeutPas:
    def test_une_relecture_qui_ne_tourne_pas_est_dite_et_le_livre_livre(
        self, monkeypatch, file_dun_livre, capsys
    ):
        """Le rapport d'un livre relu et celui d'un livre jamais relu ne
        doivent pas se ressembler."""
        code, journal, etat = lancer(monkeypatch, file_dun_livre, {"relecture": None})
        assert code == 0 and etat["status"] == "done"
        assert len(narrations(journal)) == 1
        assert "relecture impossible" in capsys.readouterr().out

    def test_sans_relecture_ne_la_lance_pas(self, monkeypatch, file_dun_livre):
        _, journal, _ = lancer(monkeypatch, file_dun_livre, {"relecture": None},
                               "--sans-relecture")
        assert not any("relire_livre.py" in c for c in journal)

    def test_un_reassemblage_qui_echoue_nest_pas_termine(self, monkeypatch, file_dun_livre):
        """Le second passage échoue : le M4B à bord est celui d'avant la
        réparation, et « terminé » ne se rejoue jamais."""
        rapport = {"segments": 40, "tronques": 1, "repares": 1, "restants": []}
        appels = {"n": 0}
        journal: list = []
        base = faux_run({"relecture": rapport}, journal)

        def run(cmd, logfile=None):
            rc, out = base(cmd, logfile)
            if "narrate_book.py" in " ".join(map(str, cmd)) and "--dry-run" not in cmd:
                appels["n"] += 1
                if appels["n"] == 2:
                    return 1, "ffmpeg a échoué"
            return rc, out

        monkeypatch.setattr(narrate_queue, "run", run)
        monkeypatch.setattr(sys, "argv", [
            "narrate_queue.py", str(file_dun_livre), "--device", "cpu",
            "--outroot", str(file_dun_livre.parent.parent / "out"), "--sans-couverture",
        ])
        narrate_queue.main()
        etat = json.loads((file_dun_livre.parent / "state.json").read_text(encoding="utf-8"))
        assert etat["livre-x"]["status"] == "failed"
        assert etat["livre-x"]["stage"] == "réassemblage"


class TestRapportRelecture:
    def test_absent_ou_illisible_vaut_none(self, tmp_path):
        assert narrate_queue.rapport_relecture(tmp_path) is None
        (tmp_path / "relecture_report.json").write_text("{pas du json", encoding="utf-8")
        assert narrate_queue.rapport_relecture(tmp_path) is None

    def test_un_rapport_sans_compte_de_segments_nest_pas_un_rapport(self, tmp_path):
        (tmp_path / "relecture_report.json").write_text("{}", encoding="utf-8")
        assert narrate_queue.rapport_relecture(tmp_path) is None

    def test_zero_segment_relu_nest_pas_une_relecture(self, tmp_path):
        """Vu sur un vrai dossier : plan et cache désaccordés, « 0 relu,
        0 tronqué », code zéro. Le rapport d'un livre sain et celui d'un livre
        jamais regardé étaient identiques."""
        (tmp_path / "relecture_report.json").write_text(
            '{"segments": 0, "tronques": 0, "repares": 0}', encoding="utf-8")
        assert narrate_queue.rapport_relecture(tmp_path) is None

    def test_les_champs_manquants_valent_zero(self, tmp_path):
        (tmp_path / "relecture_report.json").write_text('{"segments": 3}', encoding="utf-8")
        r = narrate_queue.rapport_relecture(tmp_path)
        assert (r["tronques"], r["repares"]) == (0, 0)
