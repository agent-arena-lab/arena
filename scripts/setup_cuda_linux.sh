#!/usr/bin/env bash
# One-time setup on a Linux x86_64 server with an NVIDIA GPU (e.g. IONOS Cloud GPU VM H200, Ubuntu 22.04/24.04).
#   1. checks GPU + driver   2. Python venv (HF CLI + native harness)
#   3. starts the model downloads IN THE BACKGROUND (they run while llama.cpp compiles)
#   4. CUDA toolkit (nvcc) if missing, matching the driver   5. builds llama.cpp (CUDA, GPU arch auto-detected)
#
# Usage: scripts/setup_cuda_linux.sh [--skip-models] [model ...]
#   models: default ARENA_MODELS from .env, else the models of configs/battle_h200.yaml
#   CUDA_ARCH=90 to override the detected compute capability (H100/H200 = 90)
source "$(dirname "$0")/common.sh"

SKIP_MODELS=0; models=()
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    -*) die "unknown option $a" ;;
    *) models+=("$a") ;;
  esac
done
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"

# --- 1. GPU ------------------------------------------------------------------
[[ "$(uname -m)" == "x86_64" ]] || die "this script is for x86_64 (Jetson: scripts/setup_jetson.sh)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver. Use a GPU image with driver, or 'sudo ubuntu-drivers install' + reboot."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[gpu] /' >&2
DRV_CUDA="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' | head -n1)"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')"
CUDA_ARCH="${CUDA_ARCH:-${CC:-90}}"
log "driver supports CUDA $DRV_CUDA, building for sm_$CUDA_ARCH"

log "Installing build deps ..."
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential python3-venv python3-pip curl tmux >/dev/null

# --- 2. venv -------------------------------------------------------------------
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then python3 -m venv "$ROOT_DIR/.venv"; fi
"$ROOT_DIR/.venv/bin/pip" install -q --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -q huggingface_hub
log "venv ready: $ROOT_DIR/.venv"

# --- 3. model downloads in the background --------------------------------------
DL_PID=""
if [[ $SKIP_MODELS -eq 0 ]]; then
  if [[ ${#models[@]} -eq 0 && -n "${ARENA_MODELS:-}" ]]; then read -r -a models <<<"$ARENA_MODELS"; fi
  if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/configs/battle_h200.yaml"); fi
  DL_LOG="$LOG_DIR/download_$(date +%Y%m%d-%H%M%S).log"
  log "downloading in background: ${models[*]} -> $DL_LOG"
  "$ROOT_DIR/scripts/download_models.sh" "${models[@]}" >"$DL_LOG" 2>&1 &
  DL_PID=$!
fi

# --- 4. CUDA toolkit -----------------------------------------------------------
if ! command -v nvcc >/dev/null; then
  # shellcheck disable=SC1091
  . /etc/os-release
  dist="ubuntu${VERSION_ID//./}"
  log "nvcc missing -> CUDA toolkit from the NVIDIA apt repo ($dist)"
  curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$dist/x86_64/cuda-keyring_1.1-1_all.deb"
  $SUDO dpkg -i /tmp/cuda-keyring.deb >/dev/null
  $SUDO apt-get update -qq
  # newest toolkit that the installed driver supports (toolkit newer than driver -> runtime error)
  pkg=""
  for cand in $(apt-cache pkgnames cuda-toolkit- | grep -E '^cuda-toolkit-[0-9]+-[0-9]+$' | sort -V -r); do
    v="$(echo "$cand" | sed -E 's/cuda-toolkit-([0-9]+)-([0-9]+)/\1.\2/')"
    if [[ "$(printf '%s\n%s\n' "$v" "$DRV_CUDA" | sort -V | head -n1)" == "$v" ]]; then pkg="$cand"; break; fi
  done
  [[ -n "$pkg" ]] || die "no cuda-toolkit-X-Y <= $DRV_CUDA in the NVIDIA repo"
  log "installing $pkg (~3-4 GB) ..."
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" >/dev/null
  echo "$pkg" >"$STATE_DIR/cuda_pkg"
fi
command -v nvcc >/dev/null || die "nvcc still not found (expected /usr/local/cuda/bin)"
log "CUDA: $(nvcc --version | tail -n1)"

# --- 5. llama.cpp ----------------------------------------------------------------
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"   # pin a tag (e.g. b7xxx) for reproducibility
if [[ ! -d "$LLAMA_DIR/.git" ]]; then git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"; fi
git -C "$LLAMA_DIR" fetch --tags -q
git -C "$LLAMA_DIR" checkout -q "$LLAMA_CPP_REF"
[[ "$LLAMA_CPP_REF" == "master" ]] && git -C "$LLAMA_DIR" pull -q
log "llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD) – building (5-10 min) ..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF >/dev/null
cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server >/dev/null
"$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -n2 || true

# --- harness (native, no Docker needed on a throwaway VM) ------------------------
"$ROOT_DIR/.venv/bin/pip" install -q -e "$ROOT_DIR[analysis,dev]"

if [[ -n "$DL_PID" ]]; then
  log "waiting for model downloads ..."
  wait "$DL_PID" || { tail -n 20 "$DL_LOG" >&2; die "model download failed (see $DL_LOG)"; }
  log "models ready"
fi
log "Setup done. Next: tmux new -s arena   then   scripts/battle.sh"
