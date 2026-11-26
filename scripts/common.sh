#!/usr/bin/env bash
# Shared settings for all scripts. Override via environment or a local .env file.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ROOT_DIR/.env" ]] && set -a && source "$ROOT_DIR/.env" && set +a

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$LLAMA_DIR/build/bin/llama-quantize}"
# Mac (brew install llama.cpp) or any llama-server on PATH as fallback
if [[ ! -x "$LLAMA_SERVER" ]] && command -v llama-server >/dev/null; then LLAMA_SERVER="$(command -v llama-server)"; fi
if [[ ! -x "$LLAMA_QUANTIZE" ]] && command -v llama-quantize >/dev/null; then LLAMA_QUANTIZE="$(command -v llama-quantize)"; fi
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_BASE_PORT="${LLM_BASE_PORT:-8081}"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.state}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
LOG_DIR="${LOG_DIR:-$RUNS_DIR/_server_logs}"
# docker | native  (Harness)
HARNESS_MODE="${HARNESS_MODE:-docker}"
# native | docker  (llama-server). docker braucht nvidia-container-toolkit; Miet-Container meist nur native.
SERVER_MODE="${SERVER_MODE:-native}"
# self = Originalgewichte laden und selbst identisch quantisieren (saubere Paare) | prebuilt = fertige GGUFs
ARENA_QUANT="${ARENA_QUANT:-self}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export PATH="/usr/local/cuda/bin:$PATH"

log() { printf '\033[1;34m[arena]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[arena] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

load_model_env() {
  local name="$1" f="$ROOT_DIR/models/$1.env"
  [[ -f "$f" ]] || die "unknown model '$name' (no $f)"
  unset ALIAS SRC_REPO GGUF_REPO GGUF_INCLUDE TYPE SIZE_GB MIN_RAM_GB EXTRA_ARGS NOTE
  # shellcheck disable=SC1090
  source "$f"
  MODEL_PATH="$(find "$MODELS_DIR/$ALIAS" -maxdepth 3 -name "*${QUANT_TYPE}*.gguf" ! -name "*mmproj*" 2>/dev/null | sort | head -n1 || true)"
}

all_models() { for f in "$ROOT_DIR"/models/*.env; do basename "$f" .env; done; }

# Model aliases of a config (models: as list or as mapping)
config_models() {
  awk '/^models:/{m=1; next} m && /^[^ #-]/{m=0} m && /^  - /{sub(/^  - /,""); sub(/[ #].*/,""); print; next}
       m && /^  [A-Za-z0-9._-]+:/{sub(/^  /,""); sub(/:.*/,""); print}' "$1"
}

# A value from a config (thinking.budget, thinking.ctx_per_slot, ...) without Python
config_value() {  # $1 = config, $2 = section, $3 = key, $4 = default
  awk -v s="$2" -v k="$3" '
    $0 ~ "^"s":" { insec=1; if (match($0, k": *-?[0-9]+")) { v=substr($0,RSTART,RLENGTH); sub(/.*: */,"",v); print v; exit } next }
    insec && /^[^ ]/ { insec=0 }
    insec && $0 ~ "^ +"k":" { v=$0; sub(/^ +[a-z_]+: */,"",v); sub(/ *#.*/,"",v); print v; exit }' "$1" | head -n1 | grep . || echo "$4"
}

# Portable SHA-256 (Linux: sha256sum, macOS: shasum)
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

ram_gb() {
  if [[ -r /proc/meminfo ]]; then awk '/MemTotal/{printf "%d", $2/1024/1024 + 0.5}' /proc/meminfo
  else echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 )); fi
}

# --- Hardware -----------------------------------------------------------------------------------
# Sets HW_PROFILE, HW_NAME, CUDA_ARCH, VRAM_GB, N_GPUS, GPU_IDS, MODELS_PER_GPU, GAMES_PER_SERVER, KV_TYPE,
# CTX_PER_SLOT, MIN_TOK_S. Override with ARENA_HARDWARE=<profile> or single variables in .env.
detect_hardware() {
  local profile="${ARENA_HARDWARE:-}" name="" mem="" cap=""
  N_GPUS=1; GPU_IDS="0"
  if [[ -z "$profile" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then profile=mac
    elif [[ -f /etc/nv_tegra_release ]]; then profile=jetson
    elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
      name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
      case "$name" in
        *"RTX PRO 6000"*) profile=rtxpro6000 ;;
        *H200*) profile=h200 ;;
        *H100*) profile=h100 ;;
        *A6000*) profile=a6000 ;;
        *3090*) profile=rtx3090 ;;
        *) profile=generic ;;
      esac
    else
      profile=generic
    fi
  fi
  [[ -f "$ROOT_DIR/hardware/$profile.env" ]] || die "unknown hardware profile '$profile' (hardware/*.env)"
  local keep_mpg="${MODELS_PER_GPU:-}" keep_gps="${GAMES_PER_SERVER:-}" keep_kv="${KV_TYPE:-}" keep_ctx="${CTX_PER_SLOT:-}"
  # shellcheck disable=SC1090
  source "$ROOT_DIR/hardware/$profile.env"
  HW_PROFILE="$profile"
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && [[ "$profile" != jetson ]]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
    N_GPUS="$(tr ',' '\n' <<<"$GPU_IDS" | grep -c .)"
    mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' || true)"
    [[ -n "$cap" && "$cap" =~ ^[0-9]+$ ]] && CUDA_ARCH="$cap"
    [[ -n "$mem" ]] && VRAM_GB=$(( mem / 1024 ))
    if [[ "$profile" == generic ]]; then  # derive from memory
      if (( VRAM_GB >= 130 )); then MODELS_PER_GPU=4; GAMES_PER_SERVER=2
      elif (( VRAM_GB >= 75 )); then MODELS_PER_GPU=2; GAMES_PER_SERVER=2
      elif (( VRAM_GB >= 40 )); then MODELS_PER_GPU=1; GAMES_PER_SERVER=2
      else MODELS_PER_GPU=1; GAMES_PER_SERVER=1; KV_TYPE=q8_0; CTX_PER_SLOT=20480; fi
    fi
  elif [[ "$profile" == mac || "$profile" == jetson ]]; then
    VRAM_GB="$(ram_gb)"
  fi
  # explicit overrides win
  MODELS_PER_GPU="${keep_mpg:-$MODELS_PER_GPU}"; GAMES_PER_SERVER="${keep_gps:-$GAMES_PER_SERVER}"
  KV_TYPE="${keep_kv:-$KV_TYPE}"; CTX_PER_SLOT="${ARENA_CTX_PER_SLOT:-${keep_ctx:-$CTX_PER_SLOT}}"
  export HW_PROFILE HW_NAME CUDA_ARCH VRAM_GB N_GPUS GPU_IDS MODELS_PER_GPU GAMES_PER_SERVER KV_TYPE CTX_PER_SLOT MIN_TOK_S
}

hardware_summary() {
  printf 'Hardware: %s [%s] | GPUs: %s (%s) | VRAM/GPU: %s GB | CUDA-Arch: %s | Modelle/GPU: %s | Spiele/Server: %s | KV: %s | Kontext/Slot: %s | GPU-Check: >= %s tok/s\n' \
    "$HW_NAME" "$HW_PROFILE" "$N_GPUS" "$GPU_IDS" "$VRAM_GB" "${CUDA_ARCH:-–}" "$MODELS_PER_GPU" "$GAMES_PER_SERVER" \
    "$KV_TYPE" "$CTX_PER_SLOT" "$MIN_TOK_S"
}

harness() {
  # Runs the harness CLI either in the Docker container (host network) or in the local venv.
  if [[ "$HARNESS_MODE" == "docker" ]]; then
    (cd "$ROOT_DIR" && docker compose -f docker/docker-compose.yml run --rm \
      -e ARENA_MIN_TOK_S="${MIN_TOK_S:-10}" -e ARENA_GAMES_PER_SERVER="${GAMES_PER_SERVER:-1}" \
      -e ARENA_ENDPOINTS="${ARENA_ENDPOINTS:-}" harness "$@")
  else
    ARENA_MIN_TOK_S="${MIN_TOK_S:-10}" ARENA_GAMES_PER_SERVER="${GAMES_PER_SERVER:-1}" \
      "$ROOT_DIR/.venv/bin/arena" "$@"
  fi
}
