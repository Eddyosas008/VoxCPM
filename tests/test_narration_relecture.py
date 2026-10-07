"""Tests de narration.relecture — la portée alignée, et le verdict qu'on en tire.

Le contrôle qualité ne voit pas une phrase coupée en deux. La relecture la voit
en demandant jusqu'où dans le texte la transcription est allée. Ce qui est
testé ici, c'est que cette mesure résiste à ce qui a piégé les autres : la
réécriture des nombres, un mot répété en fin de texte, les paraphrases du
milieu — et qu'elle ne condamne pas un segment pour un dernier mot avalé.
"""
from __future__ import annotations

import numpy as np
import pytest

from narration import cache as cache_tools
from narration import relecture, repair

SR = 24000

PHRASE = ("La mémoire de travail retient quelques éléments à la fois, "
          "et la fatigue en réduit encore la capacité.")


# --------------------------------------------------------------------------
# Jetons
# --------------------------------------------------------------------------


def test_normaliser_plie_accents_casse_et_lexique():
    # « çe » est l'entrée la plus fréquente du lexique ; elle doit redevenir « ce ».
    assert relecture.normaliser("Çe livre, écouté") == ["ce", "livre", "ecoute"]


def test_normaliser_recolle_les_sigles_epeles():
    # Le lexique écrit « A C E » pour forcer l'épellation ; Whisper rend « ACE ».
    assert relecture.normaliser("le score A C E") == relecture.normaliser("le score ACE")


def test_normaliser_ignore_les_nombres_des_deux_cotes():
    """Le normaliseur écrit « deux mille vingt », Whisper « 2020 » : ils ne
    peuvent pas s'aligner, donc ils ne comptent pas — sinon toute phrase qui
    finit par une date semblerait tronquée."""
    assert relecture.normaliser("né en deux mille vingt") == ["ne", "en"]
    assert relecture.normaliser("né en 2020") == ["ne", "en"]


# --------------------------------------------------------------------------
# Alignement
# --------------------------------------------------------------------------


def test_une_transcription_complete_a_une_portee_de_un():
    a = relecture.aligner(relecture.normaliser(PHRASE), relecture.normaliser(PHRASE))
    assert a.portee == 1.0 and a.couverture == 1.0 and a.manquants_en_fin == 0


def test_une_transcription_tronquee_sarrete_ou_elle_sarrete():
    source = relecture.normaliser(PHRASE)
    entendu = source[:7]
    a = relecture.aligner(source, entendu)
    assert a.dernier == 7
    assert a.manquants_en_fin == len(source) - 7


def test_une_paraphrase_au_milieu_ne_reduit_pas_la_portee():
    """Whisper reformule, supprime une hésitation, change un mot : la portée
    regarde où la transcription *finit*, pas si chaque mot y est."""
    source = relecture.normaliser(PHRASE)
    entendu = source[:3] + ["des", "trucs"] + source[6:]
    a = relecture.aligner(source, entendu)
    assert a.portee == 1.0
    assert a.couverture < 1.0


def test_un_mot_repete_en_fin_de_texte_ne_gonfle_pas_la_portee():
    """Le piège mesuré : un segment coupé à 16 % rendait 92 % de « couverture »
    parce que son dernier mot entendu revenait en fin de texte. L'alignement
    ordonné apparie au plus tôt, donc il ne tombe pas dedans."""
    source = "le chat dort sur le tapis et le chien dort".split()
    entendu = "le chat dort".split()
    a = relecture.aligner(source, entendu)
    assert a.dernier == 3  # pas 10


def test_un_nom_propre_orthographie_autrement_saligne_quand_meme():
    source = relecture.normaliser("la méthode Filliozat pour les parents")
    entendu = relecture.normaliser("la méthode Filiozat pour les parents")
    assert relecture.aligner(source, entendu).couverture == 1.0


def test_deux_mots_courts_differents_ne_sont_pas_semblables():
    assert relecture.aligner(["pas"], ["par"]).alignes == 0


def test_alignement_vide():
    assert relecture.aligner([], ["a"]).portee == 1.0
    assert relecture.aligner(["a", "b"], []).portee == 0.0


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def test_un_segment_coupe_au_milieu_est_tronque():
    moitie = " ".join(PHRASE.split()[:8])
    v = relecture.juger("ch001/seg001", PHRASE, moitie)
    assert v.tronque and v.fiable
    assert "TRONQUÉ" in v.describe()


def test_un_dernier_mot_avale_nest_pas_une_troncature():
    """Whisper avale parfois le dernier mot. Ce n'est pas le segment qui
    s'arrête, c'est la transcription."""
    sans_dernier = " ".join(PHRASE.split()[:-1])
    assert not relecture.juger("x", PHRASE, sans_dernier).tronque


def test_un_segment_trop_court_nest_pas_juge():
    v = relecture.juger("x", "Bonjour à tous.", "")
    assert not v.fiable and not v.tronque
    assert "trop court" in v.describe()


def test_une_phrase_qui_finit_par_une_date_nest_pas_tronquee():
    source = "Cette étude a été publiée par l'équipe en deux mille vingt-deux."
    entendu = "Cette étude a été publiée par l'équipe en 2022."
    assert not relecture.juger("x", source, entendu).tronque


def test_une_transcription_vide_dun_long_segment_est_tronquee():
    assert relecture.juger("x", PHRASE, "").tronque


def test_les_seuils_se_cumulent():
    """Dix-huit mots dont il manque deux : la portée passe sous 0,9 mais pas
    sous 0,85, et il manque moins de trois mots — pas tronqué. Avec un seuil
    plus sévère, si."""
    source = " ".join(f"mot{i}" for i in range(18))
    entendu = " ".join(f"mot{i}" for i in range(16))
    assert not relecture.juger("x", source, entendu).tronque
    severe = relecture.Seuils(portee_min=0.95, manquants_min=2)
    assert relecture.juger("x", source, entendu, severe).tronque


def test_to_dict_porte_de_quoi_relire_a_loeil():
    d = relecture.juger("ch002/seg004", PHRASE, PHRASE).to_dict()
    assert d["segment"] == "ch002/seg004" and d["portee"] == 1.0
    assert d["source"] == PHRASE and d["entendu"] == PHRASE


# --------------------------------------------------------------------------
# Un livre entier
# --------------------------------------------------------------------------

TEXTES = [
    PHRASE,
    "Le sommeil consolide ce qui a été appris dans la journée, surtout le sommeil profond.",
    "Une courte marche après le repas améliore la glycémie et l'humeur de l'après-midi.",
]


@pytest.fixture
def plan():
    return repair.BookPlan(
        voice={"description": "voix de test", "seed": 7, "model_id": "test"},
        chapters=(
            repair.PlannedChapter(1, "Un", (repair.PlannedSegment(TEXTES[0]),
                                            repair.PlannedSegment(TEXTES[1]))),
            repair.PlannedChapter(2, "Deux", (repair.PlannedSegment(TEXTES[2]),)),
        ),
    )


@pytest.fixture
def cache(tmp_path):
    return cache_tools.ChunkCache(tmp_path / ".cache")


def remplir(plan, cache, longueurs):
    """Un audio par segment, de longueur distincte : le faux transcripteur
    retrouve le texte par la longueur, comme Whisper le retrouverait à l'oreille."""
    spec = plan.voice_spec()
    par_longueur = {}
    for chapitre in plan.chapters:
        for segment in chapitre.segments:
            n = longueurs[segment.text]
            cache.put(cache.key(segment.text, spec), SR, np.zeros(n, dtype=np.float32),
                      text=segment.text)
            par_longueur[n] = segment.text
    return par_longueur


def test_relire_livre_juge_chaque_segment_en_cache(plan, cache):
    longueurs = {t: 1000 * (i + 1) for i, t in enumerate(TEXTES)}
    par_longueur = remplir(plan, cache, longueurs)

    def transcrire(sr, wav):
        texte = par_longueur[len(wav)]
        # Le deuxième segment est relu à moitié : il s'arrête en route.
        return " ".join(texte.split()[:6]) if texte == TEXTES[1] else texte

    progres = []
    verdicts = relecture.relire_livre(plan, cache, transcrire,
                                      on_progress=lambda i, n: progres.append((i, n)))
    assert [v.label for v in verdicts] == ["ch001/seg001", "ch001/seg002", "ch002/seg001"]
    assert [v.tronque for v in verdicts] == [False, True, False]
    assert progres == [(1, 3), (2, 3), (3, 3)]


def test_relire_livre_saute_ce_qui_nest_pas_en_cache(plan, cache):
    spec = plan.voice_spec()
    cache.put(cache.key(TEXTES[2], spec), SR, np.zeros(10, dtype=np.float32), text=TEXTES[2])
    verdicts = relecture.relire_livre(plan, cache, lambda sr, wav: TEXTES[2])
    assert [v.label for v in verdicts] == ["ch002/seg001"]


def test_relire_livre_peut_se_limiter_a_des_cibles(plan, cache):
    longueurs = {t: 100 * (i + 1) for i, t in enumerate(TEXTES)}
    par_longueur = remplir(plan, cache, longueurs)
    verdicts = relecture.relire_livre(plan, cache, lambda sr, wav: par_longueur[len(wav)],
                                      cibles=["ch002/seg001"])
    assert [v.label for v in verdicts] == ["ch002/seg001"]


def test_resume_distingue_repares_et_restants(plan, cache):
    longueurs = {t: 100 * (i + 1) for i, t in enumerate(TEXTES)}
    par_longueur = remplir(plan, cache, longueurs)
    verdicts = relecture.relire_livre(
        plan, cache, lambda sr, wav: " ".join(par_longueur[len(wav)].split()[:5]))
    assert all(v.tronque for v in verdicts)
    r = relecture.resume(verdicts, repares=["ch001/seg002"])
    assert (r["segments"], r["tronques"], r["repares"]) == (3, 3, 1)
    assert r["restants"] == ["ch001/seg001", "ch002/seg001"]
    assert len(r["details"]) == 3


def test_le_module_nimporte_pas_torch_au_niveau_module():
    """Le paquet reste importable sans torch : seul le transcripteur l'importe,
    et seulement quand on l'appelle."""
    import ast
    from pathlib import Path

    arbre = ast.parse(Path(relecture.__file__).read_text(encoding="utf-8"))
    au_sommet = set()
    for noeud in arbre.body:
        if isinstance(noeud, ast.Import):
            au_sommet.update(a.name.split(".")[0] for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            au_sommet.add(noeud.module.split(".")[0])
    assert not au_sommet & {"torch", "transformers"}
