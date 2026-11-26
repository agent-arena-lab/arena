#!/usr/bin/env bash
# Start/stop llama-server for ONE model (Jetson: one model in unified memory at a time).
# Usage: scripts/serve.sh start <model> | stop | status
#   ARENA_CTX=<n>    context PER SLOT (default: CTX from models/<model>.env)
#   ARENA_SLOTS=<n>  parallel slots (default 1). Total context = ARENA_CTX * ARENA_SLOTS.
#                    For parallel runs (arena run --workers W) use 2*W: one slot per agent keeps its prompt cache.
source "$(dirname "$0")/common.sh"

PID_FILE="$STATE_DIR/llama-server.pid"
BASE="http://$LLM_HOST:$LLM_PORT"

stop() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log "stopping llama-server (pid $(cat "$PID_FILE"))"
    kill "$(cat "$PID_FILE")"
    for _ in $(seq 1 30); do kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break; sleep 1; done
    kill -9 "$(cat "$PID_FILE")" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

start() {
  local model="$1"
  load_model_env "$model"
  [[ -x "$LLAMA_SERVER" ]] || die "llama-server not found at $LLAMA_SERVER (run install.sh or scripts/setup_jetson.sh)"
  [[ -n "$MODEL_PATH" ]] || die "no GGUF for $model in $MODELS_DIR/$ALIAS (run scripts/download_models.sh $model)"
  stop
  CTX="${ARENA_CTX:-$CTX}"   # per slot, e.g. ARENA_CTX=16384 on an 8 GB Mac
  local slots="${ARENA_SLOTS:-1}"
  local total_ctx=$(( CTX * slots ))
  local logf="$LOG_DIR/${ALIAS}_$(date +%Y%m%d-%H%M%S).log"
  echo "$logf" >"$STATE_DIR/llama-server.logpath"
  log "starting $ALIAS from $MODEL_PATH (ctx $CTX x $slots slots = $total_ctx) -> $logf"
  # --parallel: 1 for sequential runs (agents of a run are computed one after another, see game.py),
  # more for parallel runs; -ngl all: full offload
  # --jinja: required for tool calling + reasoning parsing; flags verified against llama-server --help
  # shellcheck disable=SC2086
  nohup "$LLAMA_SERVER" \
    -m "$MODEL_PATH" --alias "$ALIAS" \
    --host "$LLM_HOST" --port "$LLM_PORT" \
    -c "$total_ctx" -ngl all -fa on --parallel "$slots" \
    --jinja $EXTRA_ARGS \
    >"$logf" 2>&1 &
  echo $! >"$PID_FILE"
  log "waiting for $BASE/health ..."
  for _ in $(seq 1 300); do
    if curl -sf "$BASE/health" >/dev/null; then log "ready"; return 0; fi
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || { tail -n 30 "$logf" >&2; die "llama-server exited"; }
    sleep 2
  done
  die "llama-server not ready after 10 min (see $logf)"
}

case "${1:-}" in
  start) [[ -n "${2:-}" ]] || die "usage: serve.sh start <model>"; start "$2" ;;
  stop) stop ;;
  status) curl -s "$BASE/v1/models" || echo "not running" ;;
  *) die "usage: serve.sh start <model> | stop | status" ;;
esac
