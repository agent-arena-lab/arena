#!/usr/bin/env bash
# One-time setup on the NVIDIA Jetson AGX Orin (JetPack 6.x, Ubuntu 22.04, ARM64).
#   1. checks platform   2. builds llama.cpp with CUDA (sm_87)
#   3. creates a Python venv (HF download CLI + optional native harness)
#   4. builds the harness Docker image   5. downloads the GGUF models
#
# Usage: scripts/setup_jetson.sh [--skip-models] [--skip-docker]
source "$(dirname "$0")/common.sh"

SKIP_MODELS=0; SKIP_DOCKER=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    *) die "unknown option $a" ;;
  esac
done

# --- 1. platform -----------------------------------------------------------
[[ "$(uname -m)" == "aarch64" ]] || log "WARN: not aarch64 – are you on the Jetson?"
if [[ -f /etc/nv_tegra_release ]]; then log "L4T: $(head -n1 /etc/nv_tegra_release)"; else log "WARN: /etc/nv_tegra_release missing"; fi
command -v nvcc >/dev/null || die "nvcc not found. Install JetPack (sudo apt install nvidia-jetpack) and check /usr/local/cuda/bin"
log "CUDA: $(nvcc --version | tail -n1)"
log "Tip: for stable timings run 'sudo nvpmodel -m 0 && sudo jetson_clocks' (MAXN)."

# --- 2. llama.cpp ------------------------------------------------------------
log "Installing build deps (sudo) ..."
sudo apt-get update -qq
sudo apt-get install -y -qq git cmake build-essential libcurl4-openssl-dev python3-venv python3-pip curl

LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"   # pin a tag (e.g. b6xxx) for reproducibility!
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --tags -q
git -C "$LLAMA_DIR" checkout -q "$LLAMA_CPP_REF"
[[ "$LLAMA_CPP_REF" == "master" ]] && git -C "$LLAMA_DIR" pull -q
log "llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server
"$LLAMA_SERVER" --version || true

# --- 3. venv -----------------------------------------------------------------
python3 -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/pip" install -q --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -q "huggingface_hub[cli]" -e "$ROOT_DIR[analysis,dev]"
log "venv ready: $ROOT_DIR/.venv"

# --- 4. docker image -----------------------------------------------------------
if [[ $SKIP_DOCKER -eq 0 ]]; then
  if command -v docker >/dev/null; then
    (cd "$ROOT_DIR" && docker compose -f docker/docker-compose.yml build harness)
  else
    log "WARN: docker not found -> use HARNESS_MODE=native"
  fi
fi

# --- 5. models ---------------------------------------------------------------
if [[ $SKIP_MODELS -eq 0 ]]; then
  "$ROOT_DIR/scripts/download_models.sh"
fi
log "Setup done. Next: scripts/run_all.sh configs/smoke.yaml qwen3-4b-thinking"
