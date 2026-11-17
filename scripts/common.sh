#!/usr/bin/env bash
# Shared settings for all scripts. Override via environment or a local .env file.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ROOT_DIR/.env" ]] && set -a && source "$ROOT_DIR/.env" && set +a

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8080}"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.state}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/_server_logs}"
# docker | native
HARNESS_MODE="${HARNESS_MODE:-docker}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export PATH="/usr/local/cuda/bin:$PATH"

log() { printf '\033[1;34m[coopsab]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[coopsab] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

load_model_env() {
  local name="$1" f="$ROOT_DIR/models/$1.env"
  [[ -f "$f" ]] || die "unknown model '$name' (no $f)"
  unset ALIAS HF_REPO HF_INCLUDE MODEL_GLOB CTX EXTRA_ARGS
  # shellcheck disable=SC1090
  source "$f"
  MODEL_PATH="$(find "$MODELS_DIR/$ALIAS" -maxdepth 3 -name "$MODEL_GLOB" 2>/dev/null | sort | head -n1 || true)"
}

all_models() { for f in "$ROOT_DIR"/models/*.env; do basename "$f" .env; done; }

harness() {
  # Runs the harness CLI either in the Docker container or in the local venv.
  if [[ "$HARNESS_MODE" == "docker" ]]; then
    (cd "$ROOT_DIR" && docker compose -f docker/docker-compose.yml run --rm harness "$@")
  else
    "$ROOT_DIR/.venv/bin/coopsab" "$@"
  fi
}
