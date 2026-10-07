"""Les repères MindScript deviennent des silences mesurés, jamais des mots.

Un script de méditation écrit dans MindScript Audio porte des repères entre
crochets. Trois choses seulement comptent, et chacune a son test : un repère
de pause devient la durée exacte que l'app aurait rendue ; une indication
d'interprétation disparaît sans couper la phrase qu'elle habitait ; et aucun
repère, jamais, n'atteint le moteur — un repère lu à voix haute dans une
méditation est le pire défaut possible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import cues  # noqa: E402
from narration.chunking import split_into_segments  # noqa: E402
from narration.text_en import normalize_english  # noqa: E402
from narration.text_fr import normalize_french  # noqa: E402


def seconds_of(text: str) -> float:
    match = cues.CUE_RE.search(text)
    assert match is not None, text
    return cues.cue_seconds(match)


class TestCueSeconds:
    def test_les_durees_sont_celles_de_lapp(self):
        assert seconds_of("[PAUSE:BREATH]") == 4.0
        assert seconds_of("[PAUSE:TRANSITION]") == 5.0
        assert seconds_of("[PAUSE:INTEGRATION]") == 8.0
        assert seconds_of("[PAUSE:DEEP]") == 10.0
        assert seconds_of("[RESPIRATION]") == 4.0

    def test_un_silence_porte_sa_duree(self):
        assert seconds_of("[SILENCE 6]") == 6.0
        assert seconds_of("[SILENCE 2,5]") == 2.5
        assert seconds_of("[silence 3 s]") == 3.0

    def test_une_pause_inconnue_vaut_le_defaut(self):
        """Le script a voulu un silence : il en obtient un plausible."""
        assert seconds_of("[PAUSE:SACREE]") == cues.DEFAULT_PAUSE_SECONDS

    def test_une_indication_na_pas_de_duree(self):
        assert seconds_of("[RALENTIR]") is None
        assert seconds_of("[EMPHASE]") is None
        assert seconds_of("[TON PLUS GRAVE]") is None


class TestSplitOnCues:
    def test_le_silence_suit_le_texte_qui_le_precede(self):
        runs = cues.split_on_cues("Fermez les yeux. [PAUSE:BREATH] Le souffle vient.")
        assert runs == [("Fermez les yeux.", 4.0), ("Le souffle vient.", None)]

    def test_deux_reperes_consecutifs_font_deux_silences(self):
        runs = cues.split_on_cues("Voilà. [PAUSE:BREATH] [SILENCE 6] Ensuite.")
        assert runs == [("Voilà.", 10.0), ("Ensuite.", None)]

    def test_une_indication_ne_coupe_pas_la_phrase(self):
        runs = cues.split_on_cues("Reprenez, [RALENTIR] sans forcer.")
        assert len(runs) == 1
        assert runs[0][1] is None
        assert "RALENTIR" not in runs[0][0]
        assert "sans forcer" in runs[0][0]

    def test_un_repere_avant_tout_texte_est_ignore(self):
        runs = cues.split_on_cues("[PAUSE:DEEP] Bonjour.")
        assert runs == [("Bonjour.", None)]

    def test_strip_cues_rend_le_texte_prononcable(self):
        spoken = cues.strip_cues("Un. [PAUSE:DEEP] Deux. [EMPHASE] Trois.")
        assert spoken == "Un. Deux. Trois."


class TestPreparationShield:
    def test_le_chiffre_dun_silence_nest_pas_epele(self):
        """Épelé, [SILENCE 6] deviendrait [SILENCE six] : plus un repère."""
        out = normalize_french("Respirez 2 fois. [SILENCE 6] Voilà.")
        assert "[SILENCE 6]" in out
        assert "deux fois" in out

    def test_meme_protection_en_anglais(self):
        out = normalize_english("Breathe 2 times. [SILENCE 6] There.")
        assert "[SILENCE 6]" in out

    def test_sans_repere_rien_ne_change(self):
        assert normalize_french("Il a 3 chats.") == "Il a trois chats."


class TestSegmentsAvecReperes:
    def test_le_repere_remplace_la_pause_du_profil(self):
        segments = split_into_segments("Fermez les yeux. [PAUSE:BREATH] Bien.")
        assert [s.pause_after for s in segments][0] == 4.0

    def test_un_repere_seul_entre_paragraphes_fixe_la_pause(self):
        """Le motif exact du disclaimer MindScript : [PAUSE:DEEP] sur sa
        propre ligne, entre le corps et la dernière phrase."""
        segments = split_into_segments("Le corps.\n\n[PAUSE:DEEP]\n\nLa fin.")
        assert segments[0].pause_after == 10.0
        assert segments[0].text == "Le corps."

    def test_aucun_repere_natteint_le_moteur(self):
        segments = split_into_segments(
            "Un. [PAUSE:BREATH] Deux. [RALENTIR] Trois.\n\n[SILENCE 4]\n\nQuatre."
        )
        for segment in segments:
            assert not cues.CUE_RE.search(segment.text), segment.text

    def test_sans_repere_le_plan_reste_identique(self):
        text = "Première phrase. Seconde phrase.\n\nAutre paragraphe."
        avec = split_into_segments(text)
        assert [s.text for s in avec] == [
            "Première phrase. Seconde phrase.",
            "Autre paragraphe.",
        ]
        assert avec[0].pause_after == pytest.approx(0.9)
        assert avec[1].pause_after == pytest.approx(0.9)
