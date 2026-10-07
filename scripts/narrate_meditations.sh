#!/bin/bash
# Narrer une série de méditations sans surveillance : un .txt = un MP3.
#
# À la différence d'un livre, une série de méditations n'a ni chapitres à
# assembler ni export ACX : chaque séance est son propre livrable. Ce script
# reprend donc les principes de narrate_queue.py à l'échelle de la séance —
# une séance déjà rendue est sautée, une séance qui échoue n'arrête pas les
# autres, et le cache de narrate_book.py fait reprendre une séance
# interrompue au segment près.
#
#   bash scripts/narrate_meditations.sh /workspace/meditations/serie-01
#   bash scripts/narrate_meditations.sh serie-01 "Méditation guidée (voix féminine)"
#
# Variables : VOXCPM_DEVICE (défaut cuda), MEDITATIONS_OUT (défaut
# output/meditations). Les scripts portent leurs silences en repères
# MindScript ([PAUSE:*], [SILENCE n]) — voir narration/cues.py ; les
# génériques sont coupés parce qu'une méditation commence dans le calme et
# que le disclaimer déontologique est déjà dans le texte.
set -uo pipefail

cd "$(dirname "$0")/.."

DIR=${1:?usage : narrate_meditations.sh DOSSIER [VOIX]}
VOICE=${2:-Aurore — méditation guidée}
DEVICE=${VOXCPM_DEVICE:-cuda}
OUT=${MEDITATIONS_OUT:-output/meditations}

PYTHON=python3
[ -x ./.venv/bin/python ] && PYTHON=./.venv/bin/python

shopt -s nullglob
scripts=("$DIR"/*.txt)
if [ ${#scripts[@]} -eq 0 ]; then
  echo "aucun .txt dans $DIR" >&2
  exit 1
fi

ok=0 ; echecs=0 ; sautees=0
for txt in "${scripts[@]}"; do
  slug=$(basename "$txt" .txt)
  outdir="$OUT/$slug"

  # Déjà rendue : le MP3 final existe. Le code 0 de narrate_book garantit
  # depuis « Sortir zéro doit vouloir dire qu'un livre existe » que ce
  # fichier n'apparaît que si la séance est complète.
  if compgen -G "$outdir"/*_complet.mp3 > /dev/null; then
    echo "— $slug : déjà rendue, sautée"
    sautees=$((sautees + 1))
    continue
  fi

  # « 01_retour_au_calme » → « Retour au calme »
  titre=$(echo "$slug" | sed -E 's/^[0-9]+_//; s/_/ /g')
  titre="$(echo "${titre:0:1}" | tr '[:lower:]' '[:upper:]')${titre:1}"

  echo "=== $slug (« $titre », voix : $VOICE) ==="
  if LANG=C.UTF-8 LC_ALL=C.UTF-8 "$PYTHON" scripts/narrate_book.py "$txt" \
      --voice "$VOICE" \
      --title "$titre" \
      --assemble mp3 \
      --no-credits \
      --outdir "$outdir" \
      --device "$DEVICE"; then
    ok=$((ok + 1))
  else
    echo "— $slug : ÉCHEC (code $?) — on passe à la suivante" >&2
    echecs=$((echecs + 1))
  fi
done

echo
echo "Série terminée : $ok rendue(s), $sautees sautée(s), $echecs échec(s)."
echo "Livrables : $OUT/*/*_complet.mp3"
[ "$echecs" -eq 0 ]
