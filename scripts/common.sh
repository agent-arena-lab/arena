#!/usr/bin/env bash
# Shared settings for all scripts. Override via environment or a local .env file.
# Einzige Zielhardware: 1x NVIDIA RTX PRO 6000 Blackwell (hardware/rtxpro6000.env), alles nativ (kein Docker).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ROOT_DIR/.env" ]] && set -a && source "$ROOT_DIR/.env" && set +a

# llama.cpp fest gepinnt: diese Version lief im Podtest auf der RTX PRO 6000
LLAMA_CPP_REF="${LLAMA_CPP_REF:-fa9f928f2}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_BASE_PORT="${LLM_BASE_PORT:-8081}"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.state}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
LOG_DIR="${LOG_DIR:-$RUNS_DIR/_server_logs}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
TIMINGS="${TIMINGS:-$RUNS_DIR/_system/timings.tsv}"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$(dirname "$TIMINGS")"
export PATH="/usr/local/cuda/bin:$PATH"
export PYTHONUNBUFFERED=1   # Harness-Ausgabe sofort im Terminal/Log, nicht gepuffert

log() { printf '\033[1;34m[arena %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '\033[1;31m[arena] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Dauer einer Phase messen und in runs/_system/timings.tsv festhalten:  timed <name> <befehl ...>
timed() {
  local name="$1"; shift
  local t0 rc=0; t0=$(date +%s)
  log "▶ $name"
  "$@" || rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$dt" "$rc" >>"$TIMINGS"
  log "■ $name: $(( dt / 60 )) min $(( dt % 60 )) s (rc=$rc)"
  return $rc
}

# Befehl mit Wiederholung (apt, Downloads):  retry <versuche> <befehl ...>
retry() {
  local n="$1" i; shift
  for (( i = 1; i <= n; i++ )); do
    "$@" && return 0
    (( i < n )) && { log "Versuch $i/$n fehlgeschlagen: $* – neuer Versuch in $(( 10 * i )) s"; sleep $(( 10 * i )); }
  done
  return 1
}

load_model_env() {
  local name="$1" f="$ROOT_DIR/models/$1.env"
  [[ -f "$f" ]] || die "unknown model '$name' (no $f)"
  unset ALIAS SRC_REPO GGUF_REPO GGUF_INCLUDE TYPE SIZE_GB SRC_GB MIN_RAM_GB EXTRA_ARGS NOTE
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

# Wert aus einer Config über die Harness (venv), z. B. cfg_info configs/core.yaml thinking.budget
cfg_info() { "$ROOT_DIR/.venv/bin/arena" info --config "$1" "$2"; }

sha256() { sha256sum "$@"; }

# --- Hardware: nur RTX PRO 6000 ---------------------------------------------------------------------
# Setzt HW_PROFILE, HW_NAME, CUDA_ARCH, VRAM_GB, N_GPUS, GPU_IDS, MODELS_PER_GPU, GAMES_PER_SERVER, KV_TYPE,
# CTX_PER_SLOT, MIN_TOK_S. Andere GPUs werden abgelehnt.
detect_hardware() {
  local keep_mpg="${MODELS_PER_GPU:-}" keep_gps="${GAMES_PER_SERVER:-}" keep_kv="${KV_TYPE:-}" keep_ctx="${CTX_PER_SLOT:-}"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/hardware/rtxpro6000.env"
  HW_PROFILE=rtxpro6000
  command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || die "keine NVIDIA-GPU gefunden (nvidia-smi)"
  local name; name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  [[ "$name" == *"RTX PRO 6000"* ]] || die "GPU '$name' wird nicht unterstützt – arena läuft nur auf der RTX PRO 6000"
  GPU_IDS="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n1)}"
  N_GPUS="$(tr ',' '\n' <<<"$GPU_IDS" | grep -c .)"
  MODELS_PER_GPU="${keep_mpg:-$MODELS_PER_GPU}"; GAMES_PER_SERVER="${keep_gps:-$GAMES_PER_SERVER}"
  KV_TYPE="${keep_kv:-$KV_TYPE}"; CTX_PER_SLOT="${keep_ctx:-$CTX_PER_SLOT}"
  export HW_PROFILE HW_NAME CUDA_ARCH VRAM_GB N_GPUS GPU_IDS MODELS_PER_GPU GAMES_PER_SERVER KV_TYPE CTX_PER_SLOT MIN_TOK_S
}

hardware_summary() {
  printf 'Hardware: %s | GPU: %s | Modelle gleichzeitig: %s | Spiele/Server: %s | KV: %s | Kontext/Slot: %s | GPU-Check: >= %s tok/s\n' \
    "$HW_NAME" "$GPU_IDS" "$MODELS_PER_GPU" "$GAMES_PER_SERVER" "$KV_TYPE" "$CTX_PER_SLOT" "$MIN_TOK_S"
}

harness() {
  ARENA_MIN_TOK_S="${MIN_TOK_S:-10}" ARENA_GAMES_PER_SERVER="${GAMES_PER_SERVER:-1}" \
    "$ROOT_DIR/.venv/bin/arena" "$@"
}
