#!/usr/bin/env bash
# Installiert arena auf einer Linux-Maschine mit NVIDIA RTX PRO 6000 (einzige unterstützte Hardware).
# Für Lium-Pods gibt es scripts/pod.sh (test | full), siehe README.
#
# Einzeiler (öffentliches Repo, kein Konto nötig):
#   curl -fsSL https://raw.githubusercontent.com/agent-arena-lab/arena/main/install.sh | bash
#
# Alles, was dieses Skript anlegt oder installiert, wird in ~/.arena/manifest protokolliert.
# uninstall.sh entfernt genau diese Einträge wieder.
#
# Optionen über Umgebungsvariablen:
#   ARENA_DIR=~/arena          Installationsordner (Repo, venv, Modelle, llama.cpp, runs/ liegen ALLE hier)
#   ARENA_BRANCH=main          Branch, der geklont wird
#   ARENA_SKIP_MODELS=1        Modelle nicht herunterladen
set -euo pipefail

# Alles steckt in main(): Bei `curl ... | bash` liest bash das Skript sonst stückweise aus der Pipe.
main() {

REPO="agent-arena-lab/arena"
ARENA_DIR="${ARENA_DIR:-$HOME/arena}"
ARENA_BRANCH="${ARENA_BRANCH:-main}"
ARENA_HOME="$HOME/.arena"
MANIFEST="$ARENA_HOME/manifest"

log() { printf '\033[1;34m[arena]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[arena] FEHLER:\033[0m %s\n' "$*" >&2; exit 1; }
record() { printf '%s\t%s\n' "$1" "$2" >>"$MANIFEST"; }

[[ -e "$ARENA_DIR" ]] && die "$ARENA_DIR existiert bereits. Erst uninstall.sh ausführen oder ARENA_DIR=... setzen."
[[ "$(uname -s)" == Linux ]] || die "Nur Linux mit NVIDIA RTX PRO 6000 wird unterstützt."
command -v nvidia-smi >/dev/null || die "nvidia-smi fehlt – keine NVIDIA-GPU."
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$gpu" == *"RTX PRO 6000"* ]] || die "GPU '$gpu' wird nicht unterstützt – arena läuft nur auf der RTX PRO 6000."
mkdir -p "$ARENA_HOME"
touch "$MANIFEST"

log "Klone $REPO ($ARENA_BRANCH) nach $ARENA_DIR ..."
command -v git >/dev/null || die "git fehlt (sudo apt install git)."
git clone -q -b "$ARENA_BRANCH" "https://github.com/$REPO.git" "$ARENA_DIR"
record dir "$ARENA_DIR"
touch "$ARENA_DIR/.arena-installed"   # Sicherheitsmarker: uninstall.sh löscht nur Ordner mit diesem Marker

cat >"$ARENA_DIR/.env" <<EOF
LLAMA_DIR=$ARENA_DIR/llama.cpp
MODELS_DIR=$ARENA_DIR/models
HF_HOME=$ARENA_DIR/.hf
EOF

for p in git cmake build-essential python3-venv python3-pip curl ca-certificates; do
  dpkg -s "$p" >/dev/null 2>&1 || record apt "$p"
done
log "Setup (llama.cpp gepinnt, CUDA sm_120, venv) ..."
"$ARENA_DIR/scripts/setup_gpu.sh" --skip-models

if [[ "${ARENA_SKIP_MODELS:-0}" != "1" ]]; then
  log "Modelle (fertige GGUFs, ca. 70 GB) ..."
  "$ARENA_DIR/scripts/prepare_models.sh"
fi

log "Fertig. Installiert in $ARENA_DIR (Protokoll: $MANIFEST)"
cat >&2 <<EOF

Nächste Schritte:
  cd $ARENA_DIR
  scripts/run_all.sh configs/podtest.yaml      # Testversion: alle vier Modelle, kurze Spiele
  scripts/run_all.sh configs/pilot.yaml        # Pilot ohne Thinking-Limit, info_hint an/aus (32 Spiele)
  scripts/run_all.sh configs/core.yaml         # Kerndesign (160 Spiele)
  scripts/run_all.sh configs/extra.yaml        # feste Zusatzläufe (40 Spiele)

Alles wieder entfernen (inkl. runs/ – vorher sichern!):
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/uninstall.sh | bash
EOF
}

main "$@"
