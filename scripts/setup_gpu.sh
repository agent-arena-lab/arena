#!/usr/bin/env bash
# Einmaliges Setup auf der RTX PRO 6000 (Miet-Pod, Image nvidia/cuda:12.8.1-devel-ubuntu24.04):
#   1. GPU prüfen   2. llama.cpp (gepinnt, LLAMA_CPP_REF) mit CUDA sm_120 bauen – übersprungen, wenn der Build
#   für genau diese Version schon da ist (z. B. auf dem Lium-Volume)   3. Python-venv (nur Harness + HF-CLI)
#   4. Modelle (scripts/prepare_models.sh, fertige GGUFs)
#
# Usage: scripts/setup_gpu.sh [--skip-models]
source "$(dirname "$0")/common.sh"

build_llama() {
  local stamp="$LLAMA_DIR/build/.arena_ref"
  if [[ -x "$LLAMA_SERVER" && -f "$stamp" && "$(cat "$stamp")" == "$LLAMA_CPP_REF" ]]; then
    log "llama.cpp $LLAMA_CPP_REF bereits gebaut ($LLAMA_SERVER) – übersprungen"
    return 0
  fi
  [[ -d "$LLAMA_DIR/.git" ]] || retry 3 git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  retry 3 git -C "$LLAMA_DIR" fetch -q origin
  git -C "$LLAMA_DIR" checkout -q "$LLAMA_CPP_REF"
  log "llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD), CUDA-Arch sm_$CUDA_ARCH"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server >/dev/null
  echo "$LLAMA_CPP_REF" >"$stamp"
}

apt_install() {
  retry 3 $SUDO apt-get update -qq || return 1
  retry 3 $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git cmake build-essential python3-venv python3-pip curl ca-certificates >/dev/null
}

setup_venv() {
  if [[ -x "$ROOT_DIR/.venv/bin/arena" ]] && "$ROOT_DIR/.venv/bin/python" -c 'import arena, huggingface_hub' 2>/dev/null; then
    "$ROOT_DIR/.venv/bin/pip" install -q --no-deps -e "$ROOT_DIR" && { log "venv vorhanden"; return 0; }
  fi
  if python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    python3 -m venv "$ROOT_DIR/.venv"
  else  # ältere Images (Ubuntu 22.04: Python 3.10) -> Python 3.12 über uv
    log "Python $(python3 -V 2>&1 | cut -d' ' -f2) < 3.11 -> Python 3.12 über uv"
    python3 -m pip install -q uv 2>/dev/null || python3 -m pip install -q --break-system-packages uv
    python3 -m uv venv --seed --python 3.12 "$ROOT_DIR/.venv"
  fi
  # nur Harness + Download-CLI (kein torch/transformers: es wird nicht mehr selbst quantisiert)
  retry 3 "$ROOT_DIR/.venv/bin/pip" install -q --upgrade pip
  retry 3 "$ROOT_DIR/.venv/bin/pip" install -q "huggingface_hub[cli,hf_transfer]" -e "$ROOT_DIR[analysis,dev]"
}

main() {
SKIP_MODELS=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-docker) ;;  # veraltet, ignoriert
    *) die "unknown option $a" ;;
  esac
done

detect_hardware
log "$(hardware_summary)"
command -v nvcc >/dev/null || die "nvcc nicht gefunden – Image mit CUDA-Compiler wählen (nvidia/cuda:12.8.1-devel-ubuntu24.04)"
cuda_ver="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
awk -v v="$cuda_ver" 'BEGIN{exit !(v < 12.8)}' && die "RTX PRO 6000 (sm_120) braucht CUDA >= 12.8, gefunden $cuda_ver"
log "CUDA $cuda_ver · llama.cpp $LLAMA_CPP_REF · Modelle: $MODELS_DIR · llama.cpp: $LLAMA_DIR"

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
if command -v apt-get >/dev/null; then
  timed "setup: apt" apt_install
fi
timed "setup: llama.cpp bauen" build_llama
"$LLAMA_SERVER" --version 2>&1 | head -n 2 || true
"$LLAMA_SERVER" --help 2>&1 | grep -q -- "--reasoning-budget" || die "llama.cpp $LLAMA_CPP_REF kennt --reasoning-budget nicht"
timed "setup: venv" setup_venv

if [[ $SKIP_MODELS -eq 0 ]]; then
  "$ROOT_DIR/scripts/prepare_models.sh"
fi
log "Setup fertig. Weiter: scripts/pod.sh test  bzw.  scripts/run_all.sh configs/core.yaml"
}

main "$@"
