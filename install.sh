#!/usr/bin/env bash
# Installiert arena komplett auf dem Mac (Apple Silicon), dem Jetson AGX Orin oder einer Linux-Maschine mit
# NVIDIA-GPU (Miet-GPUs: RTX 3090, RTX A6000, RTX PRO 6000, H100, H200 – Profil wird automatisch erkannt).
#
# Einzeiler (öffentliches Repo, kein Konto nötig):
#   curl -fsSL https://raw.githubusercontent.com/agent-arena-lab/arena/main/install.sh | bash
#
# Alles, was dieses Skript anlegt oder installiert, wird in ~/.arena/manifest protokolliert.
# uninstall.sh entfernt genau diese Einträge wieder -> System danach komplett sauber.
#
# Optionen über Umgebungsvariablen:
#   ARENA_DIR=~/arena          Installationsordner (Repo, venv, Modelle, llama.cpp, runs/ liegen ALLE hier)
#   ARENA_BRANCH=main          Branch, der geklont wird
#   ARENA_MODELS="..."         Modelle (Standard Mac: Qwen3-4B-Paar fertig quantisiert;
#                              Jetson/GPU: die vier Kernmodelle, selbst quantisiert aus den Originalgewichten)
#   ARENA_QUANT=self|prebuilt  selbst quantisieren (saubere Paare, Standard auf GPU) oder fertige GGUFs laden
#   ARENA_HARDWARE=h100        Hardware-Profil erzwingen (hardware/*.env)
#   ARENA_SKIP_MODELS=1        Modelle nicht herunterladen
set -euo pipefail

# Alles steckt in main(): Bei `curl ... | bash` liest bash das Skript sonst stückweise aus der Pipe,
# und Unterprogramme (brew, pip, ...) können den Rest des Skripts von stdin "wegessen".
# So wird das Skript erst vollständig eingelesen und dann ausgeführt.
main() {

REPO="agent-arena-lab/arena"
ARENA_DIR="${ARENA_DIR:-$HOME/arena}"
ARENA_BRANCH="${ARENA_BRANCH:-main}"
ARENA_HOME="$HOME/.arena"
MANIFEST="$ARENA_HOME/manifest"

log() { printf '\033[1;34m[arena]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[arena] FEHLER:\033[0m %s\n' "$*" >&2; exit 1; }
# Manifest-Zeile: <typ><TAB><wert>   typen: dir | brew | apt | docker-image
record() { printf '%s\t%s\n' "$1" "$2" >>"$MANIFEST"; }

[[ -e "$ARENA_DIR" ]] && die "$ARENA_DIR existiert bereits. Erst uninstall.sh ausführen oder ARENA_DIR=... setzen."
mkdir -p "$ARENA_HOME"
touch "$MANIFEST"

OS="$(uname -s)"
IS_NVIDIA=0
[[ "$OS" == "Linux" && ( -f /etc/nv_tegra_release || -x "$(command -v nvidia-smi 2>/dev/null)" ) ]] && IS_NVIDIA=1
[[ "$OS" == "Darwin" || $IS_NVIDIA -eq 1 ]] || die "Nur macOS, NVIDIA Jetson und Linux mit NVIDIA-GPU werden unterstützt."
if [[ "$OS" == "Darwin" ]]; then
  ARENA_MODELS="${ARENA_MODELS:-qwen3-4b-thinking qwen3-4b-thinking-abliterated}"; ARENA_QUANT="${ARENA_QUANT:-prebuilt}"
else
  ARENA_MODELS="${ARENA_MODELS:-qwen3-30b-a3b-thinking qwen3-30b-a3b-thinking-abliterated gemma-4-26b-a4b gemma-4-26b-a4b-abliterated}"
  ARENA_QUANT="${ARENA_QUANT:-self}"
fi

# --- 1. Repository ------------------------------------------------------------
log "Klone $REPO ($ARENA_BRANCH) nach $ARENA_DIR ..."
command -v git >/dev/null || die "git fehlt (Mac: xcode-select --install, Jetson: sudo apt install git)."
git clone -q -b "$ARENA_BRANCH" "https://github.com/$REPO.git" "$ARENA_DIR"
record dir "$ARENA_DIR"
touch "$ARENA_DIR/.arena-installed"   # Sicherheitsmarker: uninstall.sh löscht nur Ordner mit diesem Marker

# Alles landet im Installationsordner: llama.cpp, Modelle, Hugging-Face-Cache, venv, runs/
cat >"$ARENA_DIR/.env" <<EOF
LLAMA_DIR=$ARENA_DIR/llama.cpp
MODELS_DIR=$ARENA_DIR/models
HF_HOME=$ARENA_DIR/.hf
LLAMA_CPP_REF=${LLAMA_CPP_REF:-master}
ARENA_QUANT=$ARENA_QUANT
EOF
[[ -n "${ARENA_HARDWARE:-}" ]] && echo "ARENA_HARDWARE=$ARENA_HARDWARE" >>"$ARENA_DIR/.env"

# --- 2. Plattform -------------------------------------------------------------
if [[ "$OS" == "Darwin" ]]; then
  command -v brew >/dev/null || die "Homebrew fehlt (https://brew.sh)."
  if ! command -v llama-server >/dev/null; then
    log "Installiere llama.cpp (brew, mit Metal) ..."
    brew install llama.cpp
    record brew llama.cpp
  fi
  PY=""
  for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && { PY="$(command -v "$c")"; break; }; done
  if [[ -z "$PY" ]]; then
    log "Installiere Python 3.12 (brew) ..."
    brew install python@3.12
    record brew python@3.12
    PY="$(brew --prefix)/bin/python3.12"
  fi
  log "Python-Umgebung (.venv) ..."
  "$PY" -m venv "$ARENA_DIR/.venv"
  "$ARENA_DIR/.venv/bin/pip" install -q --upgrade pip
  "$ARENA_DIR/.venv/bin/pip" install -q "huggingface_hub[cli]" -e "$ARENA_DIR[analysis,dev]"
  # Auf dem Mac läuft die Harness nativ: Docker Desktop hat keinen GPU-Zugriff und kein echtes Host-Netz.
  {
    echo "HARNESS_MODE=native"
    echo "LLAMA_SERVER=$(command -v llama-server)"
  } >>"$ARENA_DIR/.env"
else
  # Jetson/GPU: apt-Pakete merken, die VORHER fehlten (nur diese entfernt uninstall.sh wieder)
  for p in git cmake build-essential libcurl4-openssl-dev python3-venv python3-pip curl pciutils; do
    dpkg -s "$p" >/dev/null 2>&1 || record apt "$p"
  done
  command -v docker >/dev/null && docker info >/dev/null 2>&1 && record docker-image "arena-harness:latest"
  log "GPU-Setup (Hardware erkennen, llama.cpp mit CUDA, venv, ggf. Docker-Image) ..."
  "$ARENA_DIR/scripts/setup_gpu.sh" --skip-models
fi

# --- 3. Modelle ---------------------------------------------------------------
if [[ "${ARENA_SKIP_MODELS:-0}" != "1" ]]; then
  log "Modelle ($ARENA_QUANT): $ARENA_MODELS"
  # shellcheck disable=SC2086
  ARENA_QUANT="$ARENA_QUANT" "$ARENA_DIR/scripts/prepare_models.sh" $ARENA_MODELS
fi

log "Fertig. Installiert in $ARENA_DIR (Protokoll: $MANIFEST)"
cat >&2 <<EOF

Nächste Schritte:
  cd $ARENA_DIR
  scripts/run_all.sh configs/smoke.yaml        # kurzer Funktionstest (Qwen3-4B)
  scripts/run_all.sh configs/pilot.yaml        # Pilot ohne Thinking-Limit (16 Spiele)
  scripts/run_all.sh configs/core.yaml         # Kerndesign (160 Spiele)
  scripts/run_all.sh configs/extra.yaml        # feste Zusatzläufe (40 Spiele)

Alles wieder entfernen (inkl. runs/ – vorher sichern!):
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/uninstall.sh | bash
EOF
}

main "$@"
