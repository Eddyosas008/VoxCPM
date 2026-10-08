"""Tests de narration.mindscript — le pont vers MindScript Audio, sans serveur.

L'ouvreur urllib est remplacé par un faux qui enregistre chaque requête et
rend des réponses préparées. Ce qui est testé : quelle route est appelée
selon la façon d'entrer (clé ou session), ce que portent les en-têtes, comment
une réponse est déballée, et la forme sous laquelle un script arrive à la
narration.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from narration import mindscript


class Reponse:
    def __init__(self, corps: bytes, status: int = 200) -> None:
        self._corps, self.status = corps, status

    def read(self) -> bytes:
        return self._corps

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FauxOuvreur:
    """Rend, pour chaque (méthode, chemin), la réponse préparée."""

    def __init__(self, reponses: dict) -> None:
        self.reponses = reponses
        self.requetes: list = []

    def open(self, requete, timeout=None):
        chemin = requete.full_url.split("://", 1)[1].split("/", 1)[1]
        cle = (requete.get_method(), "/" + chemin)
        self.requetes.append((requete, requete.data))
        reponse = self.reponses.get(cle)
        if reponse is None:
            raise urllib.error.HTTPError(requete.full_url, 404, "Not Found", {},
                                         io.BytesIO(b'{"message":"Script not found"}'))
        if isinstance(reponse, int):
            raise urllib.error.HTTPError(requete.full_url, reponse, "err", {},
                                         io.BytesIO(b'{"message":"refus"}'))
        return Reponse(json.dumps(reponse).encode("utf-8"))


SCRIPT_FR = {
    "id": "3f2a9c1e-aaaa-bbbb-cccc-000000000001",
    "title": "Retour au calme",
    "content": "Bienvenue. [PAUSE:BREATH] Respirez.\r\n\r\n\r\n\r\nLaissez aller.   \n",
    "language": "fr", "estimatedMinutes": 7, "wordCount": 120,
    "updatedAt": "2026-10-05T10:00:00Z", "projectId": "p1",
    "configJson": json.dumps({"practiceType": "meditation", "language": "fr"}),
}
SCRIPT_EN = {
    "id": "9b8c7d6e-aaaa-bbbb-cccc-000000000002",
    "title": "Nature's Embrace: A Journey",
    "content": "Welcome. [PAUSE] Breathe.",
    "language": "en", "updatedAt": "2026-09-01T10:00:00Z",
}


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------


def test_un_script_se_lit_depuis_la_charge_de_lapp():
    s = mindscript.Script.from_payload(SCRIPT_FR)
    assert s.id.startswith("3f2a9c1e") and s.title == "Retour au calme"
    assert s.language == "fr" and s.estimated_minutes == 7 and s.word_count == 120
    assert s.config["practiceType"] == "meditation"
    assert s.project_id == "p1"


def test_la_langue_vient_de_la_config_si_le_champ_manque():
    s = mindscript.Script.from_payload({"id": "x", "title": "t", "content": "c",
                                        "configJson": '{"language": "EN"}'})
    assert s.language == "en"


def test_le_texte_est_range_sans_etre_reecrit():
    texte = mindscript.Script.from_payload(SCRIPT_FR).texte()
    assert texte == "Bienvenue. [PAUSE:BREATH] Respirez.\n\nLaissez aller.\n"


def test_lempreinte_ne_bouge_que_si_le_texte_bouge():
    a = mindscript.Script.from_payload(SCRIPT_FR)
    b = mindscript.Script.from_payload({**SCRIPT_FR, "title": "Autre titre", "updatedAt": "2027"})
    c = mindscript.Script.from_payload({**SCRIPT_FR, "content": "Autre texte."})
    assert a.empreinte() == b.empreinte() != c.empreinte()


def test_le_slug_est_lisible_et_unique():
    assert mindscript.Script.from_payload(SCRIPT_FR).slug() == "retour-au-calme-3f2a9c1e"
    assert mindscript.Script.from_payload(SCRIPT_EN).slug() == "nature-s-embrace-a-journey-9b8c7d6e"
    assert mindscript.slugifier("Éveil & Sérénité !") == "eveil-serenite"


def test_la_voix_par_defaut_depend_de_la_langue():
    assert mindscript.Script.from_payload(SCRIPT_FR).voix_par_defaut() == "Aurore — méditation guidée"
    assert mindscript.Script.from_payload(SCRIPT_EN).voix_par_defaut() is None


def test_selectionner_par_id_langue_et_date():
    scripts = [mindscript.Script.from_payload(SCRIPT_FR), mindscript.Script.from_payload(SCRIPT_EN)]
    assert [s.language for s in mindscript.selectionner(scripts, ids=["3f2a"])] == ["fr"]
    assert [s.language for s in mindscript.selectionner(scripts, langue="EN")] == ["en"]
    assert [s.language for s in mindscript.selectionner(scripts, depuis="2026-10-01")] == ["fr"]
    assert mindscript.selectionner(scripts, ids=["zzz"]) == []


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


def test_par_cle_les_scripts_viennent_de_lapi_v1_et_sont_deballes():
    ouvreur = FauxOuvreur({("GET", "/api/v1/scripts"): {"data": [SCRIPT_EN, SCRIPT_FR], "count": 2}})
    client = mindscript.Client("https://exemple.test/", api_key="k-123", opener=ouvreur)
    assert client.mode == "clé d'API"
    scripts = client.scripts()
    # Les plus récents d'abord.
    assert [s.language for s in scripts] == ["fr", "en"]
    requete, _ = ouvreur.requetes[0]
    assert requete.get_header("X-api-key") == "k-123"
    assert requete.full_url == "https://exemple.test/api/v1/scripts"


def test_par_session_on_se_connecte_puis_on_lit_lapi_de_lapp():
    ouvreur = FauxOuvreur({
        ("POST", "/api/login"): {"id": "u1", "email": "e@x.fr", "plan": "free"},
        ("GET", "/api/scripts"): [SCRIPT_FR],
        ("GET", "/api/scripts/" + SCRIPT_FR["id"]): SCRIPT_FR,
    })
    client = mindscript.Client("https://exemple.test", opener=ouvreur)
    assert client.mode == "aucun"
    utilisateur = client.connecter("e@x.fr", "secret")
    assert utilisateur["id"] == "u1" and client.mode == "session"
    requete, corps = ouvreur.requetes[0]
    assert json.loads(corps) == {"email": "e@x.fr", "password": "secret"}
    assert requete.get_header("Content-type") == "application/json"

    assert [s.title for s in client.scripts()] == ["Retour au calme"]
    assert client.script(SCRIPT_FR["id"]).title == "Retour au calme"
    assert all("x-api-key" not in {k.lower() for k in r.headers} for r, _ in ouvreur.requetes)


def test_une_session_prime_sur_la_cle():
    """Une clé est en lecture seule et réservée à certains plans : dès qu'une
    session est ouverte, c'est elle qui parle, et l'API de l'app qui répond."""
    ouvreur = FauxOuvreur({("POST", "/api/login"): {"id": "u1"}, ("GET", "/api/scripts"): []})
    client = mindscript.Client("https://exemple.test", api_key="k", opener=ouvreur)
    client.connecter("e", "p")
    client.scripts()
    assert ouvreur.requetes[-1][0].full_url.endswith("/api/scripts")


def test_une_erreur_http_est_dite_avec_son_code_et_son_motif():
    ouvreur = FauxOuvreur({("GET", "/api/v1/scripts"): 403})
    client = mindscript.Client("https://exemple.test", api_key="k", opener=ouvreur)
    with pytest.raises(mindscript.ErreurMindScript, match="403 : refus"):
        client.scripts()


def test_un_script_introuvable_est_une_erreur_claire():
    client = mindscript.Client("https://exemple.test", api_key="k", opener=FauxOuvreur({}))
    with pytest.raises(mindscript.ErreurMindScript, match="404"):
        client.script("inconnu")


def test_le_depot_exige_une_session(tmp_path):
    client = mindscript.Client("https://exemple.test", api_key="k", opener=FauxOuvreur({}))
    fichier = tmp_path / "x.mp3"
    fichier.write_bytes(b"ID3")
    with pytest.raises(mindscript.ErreurMindScript, match="session"):
        client.televerser_audio(fichier, "x")


def test_le_depot_envoie_un_formulaire_multipart_avec_le_fichier(tmp_path):
    ouvreur = FauxOuvreur({
        ("POST", "/api/login"): {"id": "u1"},
        ("POST", "/api/audio/assets/upload"): {"id": "asset-9", "name": "Retour au calme"},
    })
    client = mindscript.Client("https://exemple.test", opener=ouvreur)
    client.connecter("e", "p")
    fichier = tmp_path / "retour_complet.mp3"
    fichier.write_bytes(b"ID3\x00octets")

    depot = client.televerser_audio(fichier, "Retour au calme")

    assert depot["id"] == "asset-9"
    requete, corps = ouvreur.requetes[-1]
    assert requete.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert b'name="audioFile"; filename="retour_complet.mp3"' in corps
    assert b"Content-Type: audio/mpeg" in corps
    assert b"ID3\x00octets" in corps
    assert b'name="name"\r\n\r\nRetour au calme' in corps
    assert b'name="category"\r\n\r\ncustom' in corps
    assert b'name="tags"\r\n\r\nmindscript,voxcpm' in corps


def test_encoder_multipart_termine_par_la_frontiere():
    corps, content_type = mindscript.encoder_multipart({"a": "1"}, ("f", "x.bin", b"\x01\x02", "application/octet-stream"))
    frontiere = content_type.split("boundary=")[1]
    assert corps.startswith(f"--{frontiere}\r\n".encode())
    assert corps.endswith(f"\r\n--{frontiere}--\r\n".encode())
