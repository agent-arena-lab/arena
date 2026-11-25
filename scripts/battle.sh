#!/usr/bin/env bash
# Battle batch on a cloud GPU: per model llama-server with many slots, parallel runs, time budget.
# Finished runs are skipped on restart (resumable), so after an interruption simply start it again.
#
# Usage: scripts/battle.sh [config] [model ...]          (run inside tmux: survives SSH disconnects)
#   ARENA_BUDGET_MIN=60   total minutes in which NEW runs are started, split evenly across the models
#                         (runs still going at the cut-off are finished: +8-20 min per model)
#   ARENA_WORKERS=12      parallel runs per model (llama-server gets 2 slots per worker)
#   ARENA_CTX=32768       context per slot (16k thinking + history)
source "$(dirname "$0")/common.sh"

# native harness if the venv exists (cloud VM setup has no Docker image)
[[ -x "$ROOT_DIR/.venv/bin/arena" ]] && HARNESS_MODE=native

CONFIG="${1:-configs/battle_h200.yaml}"; shift || true
models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/$CONFIG"); fi
[[ ${#models[@]} -gt 0 ]] || die "no models found in $CONFIG"
WORKERS="${ARENA_WORKERS:-12}"
BUDGET="${ARENA_BUDGET_MIN:-60}"
export ARENA_SLOTS=$(( WORKERS * 2 ))
export ARENA_CTX="${ARENA_CTX:-32768}"
PER_MODEL="$(python3 -c "print(round($BUDGET / ${#models[@]}, 2))")"
EXP_NAME="$(grep -E '^name:' "$ROOT_DIR/$CONFIG" | awk '{print $2}')"
SYS_DIR="$ROOT_DIR/runs/$EXP_NAME/_system"
mkdir -p "$SYS_DIR"

GPU_PID=""
cleanup() { [[ -n "$GPU_PID" ]] && kill "$GPU_PID" 2>/dev/null; "$ROOT_DIR/scripts/serve.sh" stop; }
trap cleanup EXIT INT TERM

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host: $(hostname)"; uname -a
  nvidia-smi 2>/dev/null || true
  "$LLAMA_SERVER" --version 2>&1 | head -n 5 || true
  git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/llama.cpp commit: /' || true
  echo "workers=$WORKERS slots=$ARENA_SLOTS ctx_per_slot=$ARENA_CTX budget_total=$BUDGET per_model=$PER_MODEL"
} >"$SYS_DIR/system_$(date +%Y%m%d-%H%M%S).txt" 2>&1
# GPU utilisation every 30 s (shows whether the parallel runs actually load the GPU)
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 30 \
    >>"$SYS_DIR/gpu_usage.csv" 2>/dev/null &
  GPU_PID=$!
fi

log "models: ${models[*]} | $WORKERS parallel runs | budget $PER_MODEL min per model"
for m in "${models[@]}"; do
  load_model_env "$m"
  [[ -n "$MODEL_PATH" ]] || "$ROOT_DIR/scripts/download_models.sh" "$m"
  log "===== model: $m ====="
  "$ROOT_DIR/scripts/serve.sh" start "$m"
  harness check --config "$CONFIG" --model "$m" | tee "$SYS_DIR/check_$m.json"
  harness run --config "$CONFIG" --model "$m" --workers "$WORKERS" --budget-min "$PER_MODEL" \
    2>&1 | tee -a "$SYS_DIR/run_$m.log"
  "$ROOT_DIR/scripts/serve.sh" stop
done

log "analysis ..."
harness analyze "runs/$EXP_NAME"
log "export (archive + checksums) ..."
harness export "runs/$EXP_NAME"
log "done. Read first: runs/$EXP_NAME/_analysis/highlights.md   Download: runs/_exports/"
