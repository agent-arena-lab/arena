#!/usr/bin/env bash
# llama-server(s) starten/stoppen – passend zur erkannten Hardware (hardware/*.env).
#
#   scripts/serve.sh start <model ...>    startet die Modelle auf allen GPUs (MODELS_PER_GPU Server pro GPU;
#                                         gibt es mehr Server-Plätze als Modelle, laufen Modelle mehrfach)
#   scripts/serve.sh stop | status
#
# Pro Server: eigener Port (ab LLM_BASE_PORT=8081), --parallel = 2 x GAMES_PER_SERVER (2 Agenten pro Spiel),
# --ctx-size = Slots x CTX_PER_SLOT, --reasoning-budget explizit gesetzt (Gemma-4-Bug).
# Schreibt runs/_system/servers.json (Alias, URL, GPU, GGUF, SHA-256, Quelle, llama.cpp-Version, Flags);
# daraus liest die Harness ihre Endpunkte.
# Umgebung: REASONING_BUDGET (Standard 16384), REASONING_BUDGET_MESSAGE, SERVER_MODE=native|docker.
source "$(dirname "$0")/common.sh"

SERVERS_JSON="$RUNS_DIR/_system/servers.json"
PIDS="$STATE_DIR/servers.pids"

stop() {
  if [[ -f "$PIDS" ]]; then
    while read -r kind id; do
      if [[ "$kind" == docker ]]; then docker rm -f "$id" >/dev/null 2>&1 || true
      elif kill -0 "$id" 2>/dev/null; then
        kill "$id"
        for _ in $(seq 1 30); do kill -0 "$id" 2>/dev/null || break; sleep 1; done
        kill -9 "$id" 2>/dev/null || true
      fi
    done <"$PIDS"
    log "llama-server gestoppt"
  fi
  rm -f "$PIDS" "$SERVERS_JSON"
}

llama_version() {
  git -C "$LLAMA_DIR" rev-parse --short HEAD 2>/dev/null \
    || "$LLAMA_SERVER" --version 2>&1 | grep -m1 -oE 'version: .*' || echo unknown
}

start() {
  local models=("$@")
  [[ ${#models[@]} -gt 0 ]] || die "usage: serve.sh start <model ...>"
  detect_hardware
  log "$(hardware_summary)"
  stop
  [[ "$SERVER_MODE" == docker || -x "$LLAMA_SERVER" ]] || die "llama-server nicht gefunden ($LLAMA_SERVER) – scripts/setup_gpu.sh"

  local budget="${REASONING_BUDGET:-16384}" msg="${REASONING_BUDGET_MESSAGE:-Time is up, act now.}"
  local slots=$(( 2 * GAMES_PER_SERVER ))
  local ctx=$(( slots * CTX_PER_SLOT ))
  local help; help="$("$LLAMA_SERVER" --help 2>&1 || true)"
  IFS=',' read -r -a gpus <<<"$GPU_IDS"
  local total=$(( ${#gpus[@]} * MODELS_PER_GPU ))
  (( total >= ${#models[@]} )) || die "nur $total Server-Plätze für ${#models[@]} Modelle – run_all.sh startet in Gruppen"

  mkdir -p "$(dirname "$SERVERS_JSON")"
  local rows=() i model gpu port logf flags
  for (( i = 0; i < total; i++ )); do
    model="${models[$(( i % ${#models[@]} ))]}"
    gpu="${gpus[$(( i / MODELS_PER_GPU ))]}"
    port=$(( LLM_BASE_PORT + i ))
    load_model_env "$model"
    [[ -n "$MODEL_PATH" ]] || die "kein GGUF für $model in $MODELS_DIR/$ALIAS (scripts/prepare_models.sh $model)"
    local args=(-c "$ctx" --parallel "$slots" -ngl 999 -fa on --jinja)
    local extra=(); read -r -a extra <<<"${EXTRA_ARGS:-}"
    args+=(${extra[@]+"${extra[@]}"} --reasoning-budget "$budget")
    if grep -q -- "--reasoning-budget-message" <<<"$help" && [[ "$budget" != "-1" ]]; then
      args+=(--reasoning-budget-message "$msg")
    fi
    [[ "$KV_TYPE" != f16 ]] && args+=(--cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE")
    flags="${args[*]}"
    logf="$LOG_DIR/${ALIAS}_p${port}_$(date +%Y%m%d-%H%M%S).log"
    log "Server $((i + 1))/$total: $ALIAS auf GPU $gpu, Port $port ($flags)"
    if [[ "$SERVER_MODE" == docker ]]; then
      local cname="arena-llama-$port"
      docker run -d --rm --name "$cname" --gpus "device=$gpu" -p "$LLM_HOST:$port:$port" \
        -v "$MODELS_DIR:/models:ro" "arena-llama:${LLAMA_CPP_REF:-master}" \
        -m "/models/$ALIAS/$(basename "$MODEL_PATH")" --alias "$ALIAS" --host 0.0.0.0 --port "$port" "${args[@]}" \
        >/dev/null
      echo "docker $cname" >>"$PIDS"
      (docker logs -f "$cname" >"$logf" 2>&1 &)
    else
      CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_SERVER" -m "$MODEL_PATH" --alias "$ALIAS" \
        --host "$LLM_HOST" --port "$port" "${args[@]}" >"$logf" 2>&1 &
      echo "pid $!" >>"$PIDS"
    fi
    local sha=""; [[ -f "$MODELS_DIR/$ALIAS/SHA256SUMS" ]] && sha="$(awk '{print $1}' "$MODELS_DIR/$ALIAS/SHA256SUMS")"
    local source_json="{}"; [[ -f "$MODELS_DIR/$ALIAS/SOURCE.json" ]] && source_json="$(tr -d '\n' <"$MODELS_DIR/$ALIAS/SOURCE.json")"
    rows+=("$ALIAS|http://$LLM_HOST:$port|$gpu|$(basename "$MODEL_PATH")|$sha|$flags|$logf|$source_json")
  done

  log "warte auf /health aller Server (Laden kann einige Minuten dauern) ..."
  local r url
  for r in "${rows[@]}"; do
    url="$(cut -d'|' -f2 <<<"$r")"
    for _ in $(seq 1 450); do
      curl -sf "$url/health" >/dev/null && break
      if [[ "$SERVER_MODE" != docker ]] && ! pgrep -f -- "--port ${url##*:}" >/dev/null; then
        tail -n 30 "$(cut -d'|' -f7 <<<"$r")" >&2
        die "llama-server für $url beendet (OOM? -> GAMES_PER_SERVER / CTX_PER_SLOT / MODELS_PER_GPU senken)"
      fi
      sleep 2
    done
    curl -sf "$url/health" >/dev/null || die "llama-server $url nach 15 min nicht bereit"
  done

  printf '%s\n' "${rows[@]}" | python3 -c '
import json, sys
llama, profile, hw = sys.argv[1], sys.argv[2], sys.argv[3]
servers = []
for line in sys.stdin:
    alias, url, gpu, gguf, sha, flags, logf, src = line.rstrip("\n").split("|", 7)
    try:
        src = json.loads(src)
    except ValueError:
        src = {}
    servers.append({"alias": alias, "url": url, "gpu": gpu, "gguf": gguf, "sha256": sha or None,
                    "quant": src.get("quant"), "source": src.get("source"), "quantized_with": src.get("llama_cpp"),
                    "llama_cpp": llama, "flags": flags, "log": logf})
json.dump({"hardware_profile": profile, "hardware": hw, "servers": servers}, sys.stdout, indent=2)
' "$(llama_version)" "$HW_PROFILE" "$HW_NAME" >"$SERVERS_JSON"
  log "bereit: ${#rows[@]} Server -> $SERVERS_JSON"
}

case "${1:-}" in
  start) shift; start "$@" ;;
  stop) stop ;;
  status) [[ -f "$SERVERS_JSON" ]] && cat "$SERVERS_JSON" || echo "keine Server gestartet" ;;
  *) die "usage: serve.sh start <model ...> | stop | status" ;;
esac
