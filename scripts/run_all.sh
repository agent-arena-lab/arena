#!/usr/bin/env bash
# Full pipeline: for each model -> start llama-server -> smoke check -> run matrix -> stop.
# Resumable: finished runs (summary.json) are skipped on restart.
#
# Usage: scripts/run_all.sh [config] [model ...]
#   scripts/run_all.sh configs/smoke.yaml qwen3-4b-thinking
#   scripts/run_all.sh configs/experiment.yaml                 # models chosen with scripts/models.sh choose
#                                                               # (ARENA_MODELS in .env), else all models of the config
#   HARNESS_MODE=native scripts/run_all.sh ...                  # without Docker
source "$(dirname "$0")/common.sh"

CONFIG="${1:-configs/experiment.yaml}"; shift || true
models=("$@")
if [[ ${#models[@]} -eq 0 && -n "${ARENA_MODELS:-}" ]]; then read -r -a models <<<"$ARENA_MODELS"; log "models from .env (ARENA_MODELS): ${models[*]}"; fi
if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/$CONFIG"); fi
[[ ${#models[@]} -gt 0 ]] || die "no models found in $CONFIG"
EXP_NAME="$(grep -E '^name:' "$ROOT_DIR/$CONFIG" | awk '{print $2}')"
SYS_DIR="$ROOT_DIR/runs/$EXP_NAME/_system"
mkdir -p "$SYS_DIR"

trap '"$ROOT_DIR/scripts/serve.sh" stop' EXIT INT TERM

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host: $(hostname)"; uname -a
  cat /etc/nv_tegra_release 2>/dev/null || true
  (nvpmodel -q 2>/dev/null || sudo -n nvpmodel -q 2>/dev/null) || echo "nvpmodel: n/a"
  "$LLAMA_SERVER" --version 2>&1 | head -n 5 || true
  git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/llama.cpp commit: /' || true
  free -h 2>/dev/null || sysctl -n hw.memsize 2>/dev/null | sed 's/^/hw.memsize: /' || true
} >"$SYS_DIR/system_$(date +%Y%m%d-%H%M%S).txt" 2>&1

for m in "${models[@]}"; do
  log "===== model: $m ====="
  "$ROOT_DIR/scripts/serve.sh" start "$m"
  harness check --config "$CONFIG" --model "$m" | tee "$SYS_DIR/check_$m.json"
  harness run --config "$CONFIG" --model "$m"
  "$ROOT_DIR/scripts/serve.sh" stop
done

log "analysis ..."
harness analyze "runs/$EXP_NAME"
log "export (Archiv + Prüfsummen) ..."
harness export "runs/$EXP_NAME"
log "done: runs/$EXP_NAME/_analysis, Export in runs/_exports/"
