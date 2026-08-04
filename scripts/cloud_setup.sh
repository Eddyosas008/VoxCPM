#!/usr/bin/env bash
# Provision any Ubuntu machine to narrate — a VPS, a rented GPU box, anything.
#
# The same script serves both because the only thing that differs between them
# is which PyTorch wheel to fetch, and that is decided here by looking for a
# GPU rather than by asking. Run it twice and it changes nothing the second
# time: every step checks before it acts.
#
#   curl -fsSL https://raw.githubusercontent.com/Eddyosas008/VoxCPM/claude/repo-analysis-improvement-dg0ies/scripts/cloud_setup.sh | bash
#
# or, once the repository is already there:
#
#   bash scripts/cloud_setup.sh
#
# Environment:
#   VOXCPM_DIR     where to install            (default: ~/voxcpm)
#   VOXCPM_BRANCH  branch to check out         (default: claude/repo-analysis-improvement-dg0ies)
#   VOXCPM_REPO    repository to clone         (default: this fork)
#   SKIP_MODEL=1   do not pre-download the model
set -euo pipefail

DIR="${VOXCPM_DIR:-$HOME/voxcpm}"
BRANCH="${VOXCPM_BRANCH:-claude/repo-analysis-improvement-dg0ies}"
REPO="${VOXCPM_REPO:-https://github.com/Eddyosas008/VoxCPM.git}"

say() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

# --- What are we on? -------------------------------------------------------
CORES="$(nproc)"
RAM_MB="$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    DEVICE="cuda"
else
    GPU=""
    # The CPU wheels are a fraction of the size of the CUDA ones, and on a box
    # without a GPU the CUDA extras are several gigabytes of dead weight.
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    DEVICE="cpu"
fi

say "Machine : ${CORES} cœur(s), ${RAM_MB} Mo de RAM, ${GPU:-pas de GPU} → device=${DEVICE}"

if [ "$DEVICE" = "cpu" ] && [ "$RAM_MB" -lt 12000 ]; then
    # float32 weights need about 8.7 GB resident, and the load is where it dies.
    echo "    RAM limitée : lancez la narration avec VOXCPM_CPU_DTYPE=bfloat16"
    echo "    (empreinte divisée par deux, un peu plus lent — mais il faut que ça tienne)"
fi

# --- System packages -------------------------------------------------------
say "Paquets système"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git curl python3 python3-venv python3-pip ffmpeg libsndfile1

# --- The repository --------------------------------------------------------
if [ -d "$DIR/.git" ]; then
    say "Mise à jour de $DIR"
    git -C "$DIR" fetch --quiet origin "$BRANCH"
    git -C "$DIR" checkout --quiet "$BRANCH"
    git -C "$DIR" pull --quiet --ff-only origin "$BRANCH"
else
    say "Clonage dans $DIR"
    git clone --quiet --branch "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

# --- Python ----------------------------------------------------------------
say "Environnement Python"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel

say "PyTorch (${DEVICE})"
python - <<'PY' || pip install --quiet torch torchaudio --index-url "$TORCH_INDEX"
import sys
try:
    import torch  # noqa: F401
except ImportError:
    sys.exit(1)
PY

say "Dépendances du projet"
pip install --quiet -e .

# --- The model ------------------------------------------------------------
if [ "${SKIP_MODEL:-0}" != "1" ]; then
    say "Téléchargement du modèle (≈4,6 Go, une seule fois)"
    python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("openbmb/VoxCPM2")
print(f"    modèle dans {path}")
PY
fi

# --- Ready ----------------------------------------------------------------
say "Prêt"
cat <<EOF

  cd $DIR && source .venv/bin/activate

  # Narrer un livre, sans surveillance (survit à la déconnexion) :
  nohup ./.venv/bin/python scripts/narrate_book.py livre.epub \\
      --voice "Narrateur profond & calme" --device $DEVICE \\
      --assemble m4b --export-acx > narration.log 2>&1 &

  # Suivre :
  tail -f narration.log

  # L'interface, accessible uniquement par tunnel SSH (recommandé) :
  ./.venv/bin/python app.py --host 127.0.0.1 --port 8808 --device $DEVICE --no-denoiser
  #   puis depuis votre poste :  ssh -N -L 8808:127.0.0.1:8808 root@<ip>
  #   et ouvrez http://127.0.0.1:8808

  # Ou exposée, avec mot de passe obligatoire :
  VOXCPM_AUTH='edwin:motdepasse' ./.venv/bin/python app.py \\
      --host 0.0.0.0 --port 8808 --device $DEVICE --no-denoiser

  # Rapatrier les chapitres finis, depuis votre poste :
  rsync -avz root@<ip>:$DIR/output/book_<nom>/ ./book_<nom>/

EOF
