"""Tests de bout en bout de scripts/narrate_mindscript.py.

Le client MindScript est un faux qui rend des scripts préparés, et chaque
étape (narration, relecture) est un sous-processus remplacé par un faux qui
fabrique ce qu'elle produirait. Ce qui est testé : ce qui est écrit sur le
disque, la commande de narration, et les décisions — sauter un script à jour,
renarrer un script modifié, écarter un script sans voix, déposer le MP3.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import mindscript  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "narrate_mindscript", ROOT / "scripts" / "narrate_mindscript.py"
)
narrate_mindscript = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(narrate_mindscript)


def script(ident: str, titre: str, contenu: str, langue: str = "fr", maj: str = "2026-10-05") -> mindscript.Script:
    return mindscript.Script(id=ident, title=titre, content=contenu, language=langue,
                             estimated_minutes=5, updated_at=maj)


class FauxClient:
    def __init__(self, scripts, session=True) -> None:
        self._scripts = scripts
        self.utilisateur = {"id": "u1"} if session else None
        self.mode = "session" if session else "clé d'API"
        self.depots: list = []

    def scripts(self):
        return list(self._scripts)

    def script(self, ident):
        return next(s for s in self._scripts if s.id == ident)

    def televerser_audio(self, chemin, nom, **_):
        self.depots.append((Path(chemin), nom))
        return {"id": f"asset-{len(self.depots)}"}


def faux_run(journal: list, scenario: dict | None = None):
    scenario = scenario or {}

    def run(cmd, logfile=None):
        texte = " ".join(str(c) for c in cmd)
        journal.append(cmd)
        if "narrate_book.py" in texte:
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            if scenario.get("narration_rc", 0) == 0:
                (outdir / "chapitre_001.wav").write_bytes(b"RIFF")
                (outdir / f"{outdir.name}_complet.mp3").write_bytes(b"ID3")
            return scenario.get("narration_rc", 0), ""
        if "relire_livre.py" in texte:
            outdir = Path(cmd[2])
            rapport = scenario.get("relecture")
            if rapport is not None:
                (outdir / "relecture_report.json").write_text(json.dumps(rapport), encoding="utf-8")
                return 0, ""
            return 2, ""
        return 0, ""

    return run


def lancer(monkeypatch, tmp_path, client, *extra, scenario=None):
    journal: list = []
    monkeypatch.setattr(narrate_mindscript, "creer_client", lambda args: client)
    monkeypatch.setattr(narrate_mindscript, "run", faux_run(journal, scenario))
    monkeypatch.setattr(sys, "argv", [
        "narrate_mindscript.py", "--outdir", str(tmp_path / "ms"), "--device", "cpu", *extra,
    ])
    code = narrate_mindscript.main()
    state_path = tmp_path / "ms" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    return code, journal, state


def narrations(journal):
    return [c for c in journal if "scripts/narrate_book.py" in c]


FR = script("3f2a9c1e-1", "Retour au calme", "Bienvenue. [PAUSE:BREATH] Respirez.")
EN = script("9b8c7d6e-2", "Deep Rest", "Welcome. [PAUSE] Breathe.", langue="en")


class TestListe:
    def test_sans_cible_on_liste_et_on_ne_narre_rien(self, monkeypatch, tmp_path, capsys):
        code, journal, _ = lancer(monkeypatch, tmp_path, FauxClient([FR, EN]))
        assert code == 0 and journal == []
        sortie = capsys.readouterr().out
        assert "Retour au calme" in sortie and "Deep Rest" in sortie
        assert "Rien à narrer" in sortie

    def test_list_sarrete_la(self, monkeypatch, tmp_path, capsys):
        code, journal, _ = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--list", "--all")
        assert code == 0 and journal == []
        assert "2 script" not in capsys.readouterr().out


class TestNarration:
    def test_un_script_est_ecrit_puis_narre_dans_la_voix_de_sa_langue(
        self, monkeypatch, tmp_path
    ):
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--script", "3f2a")
        assert code == 0
        dossier = tmp_path / "ms" / "retour-au-calme-3f2a9c1e"
        assert (dossier / "script.txt").read_text(encoding="utf-8") == FR.texte()
        meta = json.loads((dossier / "script.json").read_text(encoding="utf-8"))
        assert meta["id"] == FR.id and meta["empreinte"] == FR.empreinte()

        (cmd,) = narrations(journal)
        assert cmd[cmd.index("--voice") + 1] == "Aurore — méditation guidée"
        assert cmd[cmd.index("--title") + 1] == "Retour au calme"
        assert cmd[cmd.index("--language") + 1] == "fr"
        assert "--no-credits" in cmd and "--no-cover" in cmd
        assert cmd[cmd.index("--assemble") + 1] == "mp3"
        assert cmd[cmd.index("--lexicon") + 1] == "conf/pronunciation_fr.json"
        assert state[FR.id]["status"] == "done"
        assert state[FR.id]["mp3"].endswith("_complet.mp3")

    def test_un_script_a_jour_est_saute(self, monkeypatch, tmp_path, capsys):
        lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all")
        code, journal, _ = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all")
        assert code == 0 and narrations(journal) == []
        assert "à jour, sauté" in capsys.readouterr().out

    def test_un_script_modifie_dans_lapp_est_renarre_sans_son_ancien_mp3(
        self, monkeypatch, tmp_path, capsys
    ):
        lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all")
        dossier = tmp_path / "ms" / FR.slug()
        (dossier / ".cache").mkdir()
        (dossier / ".cache" / "abc.wav").write_bytes(b"RIFF")
        ancien = next(dossier.glob("*_complet.mp3"))
        ancien.write_bytes(b"ANCIEN")

        modifie = script(FR.id, FR.title, FR.content + " Et encore une phrase.")
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([modifie]), "--all")

        assert code == 0 and len(narrations(journal)) == 1
        assert "script modifié" in capsys.readouterr().out
        assert (dossier / ".cache" / "abc.wav").exists()  # le cache survit
        assert next(dossier.glob("*_complet.mp3")).read_bytes() == b"ID3"  # pas l'ancien
        assert state[FR.id]["empreinte"] == modifie.empreinte()

    def test_force_renarre_un_script_inchange(self, monkeypatch, tmp_path):
        lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all")
        _, journal, _ = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all", "--force")
        assert len(narrations(journal)) == 1

    def test_sans_voix_pour_sa_langue_un_script_est_ecarte_et_dit(
        self, monkeypatch, tmp_path, capsys
    ):
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([EN]), "--all")
        assert code == 1 and narrations(journal) == []
        assert state[EN.id]["status"] == "failed" and state[EN.id]["stage"] == "voix"
        assert "--voice" in capsys.readouterr().out

    def test_voice_couvre_une_langue_sans_defaut(self, monkeypatch, tmp_path):
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([EN]), "--all",
                                      "--voice", "Narratrice douce & naturelle")
        assert code == 0
        (cmd,) = narrations(journal)
        assert cmd[cmd.index("--language") + 1] == "en"
        assert "--lexicon" not in cmd
        assert state[EN.id]["voice"] == "Narratrice douce & naturelle"

    def test_un_echec_narre_les_suivants(self, monkeypatch, tmp_path):
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([FR, EN]), "--all",
                                      "--voice", "V", scenario={"narration_rc": 1})
        assert code == 1 and len(narrations(journal)) == 2
        assert all(v["status"] == "failed" and v["stage"] == "narration" for v in state.values())

    def test_export_only_ecrit_sans_narrer(self, monkeypatch, tmp_path):
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all",
                                      "--export-only")
        assert code == 0 and journal == [] and state == {}
        assert (tmp_path / "ms" / FR.slug() / "script.txt").exists()

    def test_les_filtres_de_langue_et_de_date_sappliquent(self, monkeypatch, tmp_path):
        vieux = script("aaaa-3", "Vieux", "Texte.", maj="2026-01-01")
        _, journal, _ = lancer(monkeypatch, tmp_path, FauxClient([FR, EN, vieux]), "--all",
                               "--language", "fr", "--since", "2026-10-01")
        assert len(narrations(journal)) == 1


class TestRelectureEtDepot:
    def test_la_relecture_refait_le_mp3_quand_elle_a_repare(self, monkeypatch, tmp_path):
        rapport = {"segments": 30, "tronques": 1, "repares": 1, "restants": []}
        code, journal, state = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all",
                                      "--relecture", scenario={"relecture": rapport})
        assert code == 0 and len(narrations(journal)) == 2
        assert any("relire_livre.py" in " ".join(c) for c in journal)
        assert state[FR.id]["status"] == "done"

    def test_une_relecture_saine_ne_refait_rien(self, monkeypatch, tmp_path):
        rapport = {"segments": 30, "tronques": 0, "repares": 0, "restants": []}
        _, journal, _ = lancer(monkeypatch, tmp_path, FauxClient([FR]), "--all",
                               "--relecture", scenario={"relecture": rapport})
        assert len(narrations(journal)) == 1

    def test_upload_depose_le_mp3_sous_le_titre_du_script(self, monkeypatch, tmp_path):
        client = FauxClient([FR])
        code, _, state = lancer(monkeypatch, tmp_path, client, "--all", "--upload")
        assert code == 0
        (chemin, nom), = client.depots
        assert chemin.name.endswith("_complet.mp3") and nom == "Retour au calme"
        assert state[FR.id]["asset_id"] == "asset-1"

    def test_un_depot_rate_ne_defait_pas_la_narration(self, monkeypatch, tmp_path, capsys):
        client = FauxClient([FR])

        def rate(*a, **k):
            raise mindscript.ErreurMindScript("POST /api/audio/assets/upload → 413")

        client.televerser_audio = rate
        code, _, state = lancer(monkeypatch, tmp_path, client, "--all", "--upload")
        assert code == 0 and state[FR.id]["status"] == "done"
        assert "413" in state[FR.id]["upload_error"]
        assert "relancez avec --upload" in capsys.readouterr().out


class TestAcces:
    def test_sans_cle_ni_session_cest_dit(self, monkeypatch):
        args = narrate_mindscript.build_parser().parse_args([])
        args.api_key = args.email = args.password = None
        with pytest.raises(SystemExit, match="MINDSCRIPT_API_KEY"):
            narrate_mindscript.creer_client(args)

    def test_upload_par_cle_est_refuse_avant_tout_appel(self, monkeypatch):
        args = narrate_mindscript.build_parser().parse_args(["--upload", "--api-key", "k"])
        args.email = args.password = None
        with pytest.raises(SystemExit, match="session"):
            narrate_mindscript.creer_client(args)

    def test_par_session_on_se_connecte(self, monkeypatch):
        appels = []

        class Client:
            utilisateur = None

            def __init__(self, base_url, api_key=None):
                appels.append(("new", base_url, api_key))

            def connecter(self, e, p):
                appels.append(("login", e, p))
                self.utilisateur = {"id": "u"}

        monkeypatch.setattr(narrate_mindscript.mindscript, "Client", Client)
        args = narrate_mindscript.build_parser().parse_args(
            ["--base-url", "http://localhost:5000", "--email", "e@x", "--password", "p"])
        args.api_key = None
        narrate_mindscript.creer_client(args)
        assert appels == [("new", "http://localhost:5000", None), ("login", "e@x", "p")]


class TestFichierLocal:
    """Les identifiants peuvent vivre dans conf/mindscript.local.env, ignoré
    par git : ni dans une commande, ni dans une conversation."""

    def test_le_fichier_local_pose_les_variables_manquantes(self, monkeypatch, tmp_path):
        for cle in ("MINDSCRIPT_EMAIL", "MINDSCRIPT_PASSWORD", "MINDSCRIPT_API_KEY"):
            monkeypatch.delenv(cle, raising=False)
        monkeypatch.setenv("MINDSCRIPT_PASSWORD", "deja-la")
        fichier = tmp_path / "mindscript.local.env"
        fichier.write_text(
            "# commentaire\nMINDSCRIPT_EMAIL = \"e@x.fr\"\nMINDSCRIPT_PASSWORD=autre\n"
            "AUTRE_CHOSE=non\nMINDSCRIPT_API_KEY=\n", encoding="utf-8")
        charges = narrate_mindscript.charger_env_local(fichier)
        assert charges == ["MINDSCRIPT_EMAIL"]
        assert narrate_mindscript.os.environ["MINDSCRIPT_EMAIL"] == "e@x.fr"
        assert narrate_mindscript.os.environ["MINDSCRIPT_PASSWORD"] == "deja-la"  # l'environnement prime
        assert "AUTRE_CHOSE" not in narrate_mindscript.os.environ

    def test_sans_fichier_rien_ne_casse(self, tmp_path):
        assert narrate_mindscript.charger_env_local(tmp_path / "absent.env") == []

    def test_le_fichier_local_est_ignore_par_git(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "conf/mindscript.local.env" in ignore
        assert (ROOT / "conf" / "mindscript.local.env.exemple").exists()
