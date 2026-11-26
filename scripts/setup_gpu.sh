#!/usr/bin/env bash
# Einmaliges Setup auf einer NVIDIA-Maschine (Miet-GPU aus der Hardware-Tabelle oder Jetson AGX Orin).
#   1. Hardware erkennen (hardware/*.env, nvidia-smi)   2. llama.cpp mit CUDA für genau diese Architektur bauen
#   3. Python-venv (Harness, HF-CLI, Konverter für die Selbst-Quantisierung)   4. Docker-Images (optional)
#   5. Modelle bereitstellen (scripts/prepare_models.sh)
#
# Usage: scripts/setup_gpu.sh [--skip-models] [--skip-docker]
#   ARENA_HARDWARE=h100 ...      Profil erzwingen (sonst automatisch)
#   LLAMA_CPP_REF=b7xxx ...      llama.cpp fest pinnen (Reproduzierbarkeit, Gemma-4-Bug!)
#
# Miet-GPUs (vast.ai, RunPod, ...): Das Image muss CUDA mit nvcc enthalten, z. B. nvidia/cuda:12.8.1-devel-ubuntu22.04
# (RTX PRO 6000 / Blackwell braucht CUDA >= 12.8). Solche Instanzen sind meist selbst Container ohne Docker:
# dann laufen Harness und llama-server nativ (HARNESS_MODE=native, SERVER_MODE=native; macht dieses Skript automatisch).
source "$(dirname "$0")/common.sh"

main() {
SKIP_MODELS=0; SKIP_DOCKER=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    *) die "unknown option $a" ;;
  esac
done

# --- 1. Hardware ------------------------------------------------------------------------------
detect_hardware
log "$(hardware_summary)"
command -v nvcc >/dev/null || die "nvcc nicht gefunden. Jetson: sudo apt install nvidia-jetpack. Miete: CUDA-devel-Image wählen."
cuda_ver="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
log "CUDA $cuda_ver, Architektur sm_${CUDA_ARCH}"
if [[ "$CUDA_ARCH" == 120 ]] && awk -v v="$cuda_ver" 'BEGIN{exit !(v < 12.8)}'; then
  die "Blackwell (sm_120) braucht CUDA >= 12.8, gefunden $cuda_ver"
fi
df -h "$MODELS_DIR" 2>/dev/null | tail -n1 | awk '{print "[arena] freier Platz für Modelle: "$4}' >&2 || true
log "Hinweis: Selbst-Quantisierung lädt ~226 GB Originalgewichte (4 Kernmodelle) und braucht kurzzeitig ~2x SRC_GB pro Modell."

# --- 2. llama.cpp -------------------------------------------------------------------------------
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
if command -v apt-get >/dev/null; then
  log "Build-Abhängigkeiten (apt) ..."
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq git cmake build-essential libcurl4-openssl-dev python3-venv python3-pip curl pciutils
fi
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
[[ "$LLAMA_CPP_REF" == master ]] && log "WARN: llama.cpp nicht gepinnt (LLAMA_CPP_REF=master). Nach dem Pilot in .env festlegen!"
[[ -d "$LLAMA_DIR/.git" ]] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
git -C "$LLAMA_DIR" fetch --tags -q
git -C "$LLAMA_DIR" checkout -q "$LLAMA_CPP_REF"
[[ "$LLAMA_CPP_REF" == master ]] && git -C "$LLAMA_DIR" pull -q
log "llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server llama-quantize
"$LLAMA_SERVER" --version || true
"$LLAMA_SERVER" --help 2>&1 | grep -q -- "--reasoning-budget" || log "WARN: diese llama.cpp-Version kennt --reasoning-budget nicht"

# --- 3. venv --------------------------------------------------------------------------------------
if python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
  python3 -m venv "$ROOT_DIR/.venv"
else  # z. B. nvidia/cuda:*-ubuntu22.04 hat Python 3.10 -> eigenständiges Python 3.12 über uv
  log "Python $(python3 -V 2>&1 | cut -d' ' -f2) < 3.11 -> Python 3.12 über uv"
  python3 -m pip install -q uv 2>/dev/null || python3 -m pip install -q --break-system-packages uv
  python3 -m uv venv --seed --python 3.12 "$ROOT_DIR/.venv"
fi
"$ROOT_DIR/.venv/bin/pip" install -q --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -q "huggingface_hub[cli]" -e "$ROOT_DIR[analysis,dev]"
req="$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"
if [[ -f "$req" ]]; then
  log "Konverter-Abhängigkeiten (torch CPU, transformers, gguf) ..."
  "$ROOT_DIR/.venv/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r "$req"
fi
log "venv bereit: $ROOT_DIR/.venv"

# --- 4. Docker (nur wo vorhanden) ----------------------------------------------------------------
mode_line() { grep -v "^$1=" "$ROOT_DIR/.env" 2>/dev/null >"$ROOT_DIR/.env.tmp" || true; echo "$1=$2" >>"$ROOT_DIR/.env.tmp"; mv "$ROOT_DIR/.env.tmp" "$ROOT_DIR/.env"; }
if [[ $SKIP_DOCKER -eq 0 ]] && command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  (cd "$ROOT_DIR" && docker compose -f docker/docker-compose.yml build harness)
  mode_line HARNESS_MODE docker
else
  log "kein nutzbares Docker -> Harness nativ (HARNESS_MODE=native)"
  mode_line HARNESS_MODE native
fi
mode_line ARENA_HARDWARE "$HW_PROFILE"

# --- 5. Modelle --------------------------------------------------------------------------------
if [[ $SKIP_MODELS -eq 0 ]]; then
  "$ROOT_DIR/scripts/prepare_models.sh"
fi
log "Setup fertig. Weiter: scripts/run_all.sh configs/smoke.yaml  bzw.  scripts/run_all.sh configs/core.yaml"
}

main "$@"
