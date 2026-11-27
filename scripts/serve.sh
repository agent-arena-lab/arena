#!/usr/bin/env bash
# llama-server(s) auf der RTX PRO 6000 starten/stoppen.
#
#   scripts/serve.sh start <model ...>    ein Server pro Modell (max. MODELS_PER_GPU = 2, also eine Familie)
#   scripts/serve.sh stop | status
#
# Pro Server: eigener Port (ab LLM_BASE_PORT=8081), --parallel = 2 x GAMES_PER_SERVER (2 Agenten pro Spiel),
# --ctx-size = Slots x CTX_PER_SLOT, --reasoning-budget explizit gesetzt (Gemma-4-Bug).
# Gibt die Ladezeit jedes Servers bis zur Bereitschaft aus (auch in runs/_system/timings.tsv).
# Schreibt runs/_system/servers.json (Alias, URL, GGUF, SHA-256, Quelle, llama.cpp-Version, Flags, Ladezeit);
# daraus liest die Harness ihre Endpunkte.
# Umgebung: REASONING_BUDGET (Standard 16384), REASONING_BUDGET_MESSAGE (neutral, siehe configs).
source "$(dirname "$0")/common.sh"

SERVERS_JSON="$RUNS_DIR/_system/servers.json"
PIDS="$STATE_DIR/servers.pids"

stop() {
  if [[ -f "$PIDS" ]]; then
    while read -r _ id; do
      if kill -0 "$id" 2>/dev/null; then
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
  [[ -x "$LLAMA_SERVER" ]] || die "llama-server nicht gefunden ($LLAMA_SERVER) – scripts/setup_gpu.sh"

  local budget="${REASONING_BUDGET:-16384}"
  local msg="${REASONING_BUDGET_MESSAGE:-Reasoning budget reached. Choose your next tool call.}"
  local slots=$(( 2 * GAMES_PER_SERVER ))
  local ctx=$(( slots * CTX_PER_SLOT ))
  local help; help="$("$LLAMA_SERVER" --help 2>&1 || true)"
  local gpu="${GPU_IDS%%,*}"
  local total="$MODELS_PER_GPU"
  (( total >= ${#models[@]} )) || die "nur $total Server-Plätze für ${#models[@]} Modelle – run_all.sh startet in Gruppen"

  mkdir -p "$(dirname "$SERVERS_JSON")"
  local rows=() i model port logf flags
  for (( i = 0; i < ${#models[@]}; i++ )); do
    model="${models[$i]}"
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
    log "Server $((i + 1))/${#models[@]}: $ALIAS auf GPU $gpu, Port $port ($flags)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_SERVER" -m "$MODEL_PATH" --alias "$ALIAS" \
      --host "$LLM_HOST" --port "$port" "${args[@]}" >"$logf" 2>&1 &
    echo "pid $!" >>"$PIDS"
    local sha=""; [[ -f "$MODELS_DIR/$ALIAS/SHA256SUMS" ]] && sha="$(awk '{print $1}' "$MODELS_DIR/$ALIAS/SHA256SUMS")"
    local source_json="{}"; [[ -f "$MODELS_DIR/$ALIAS/SOURCE.json" ]] && source_json="$(tr -d '\n' <"$MODELS_DIR/$ALIAS/SOURCE.json")"
    rows+=("$ALIAS|http://$LLM_HOST:$port|$gpu|$(basename "$MODEL_PATH")|$sha|$flags|$logf|$(date +%s)|$source_json")
  done

  log "warte auf /health aller Server ..."
  local r url t0 alias dt out=()
  for r in "${rows[@]}"; do
    alias="$(cut -d'|' -f1 <<<"$r")"; url="$(cut -d'|' -f2 <<<"$r")"; t0="$(cut -d'|' -f8 <<<"$r")"
    for _ in $(seq 1 450); do
      curl -sf "$url/health" >/dev/null && break
      if ! pgrep -f -- "--port ${url##*:}" >/dev/null; then
        tail -n 30 "$(cut -d'|' -f7 <<<"$r")" >&2
        die "llama-server für $url beendet (OOM? -> GAMES_PER_SERVER oder CTX_PER_SLOT senken)"
      fi
      sleep 1
    done
    curl -sf "$url/health" >/dev/null || die "llama-server $url nach 7,5 min nicht bereit"
    dt=$(( $(date +%s) - t0 ))
    log "$alias bereit nach $dt s (Laden bis /health)"
    printf '%s\t%s\t%s\t0\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "server laden: $alias" "$dt" >>"$TIMINGS"
    out+=("$r|$dt")
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[arena] GPU-Speicher: /' >&2 || true

  printf '%s\n' "${out[@]}" | python3 -c '
import json, sys
llama, profile, hw = sys.argv[1], sys.argv[2], sys.argv[3]
servers = []
for line in sys.stdin:
    alias, url, gpu, gguf, sha, flags, logf, _t0, rest = line.rstrip("\n").split("|", 8)
    src, _, load_s = rest.rpartition("|")
    try:
        src = json.loads(src)
    except ValueError:
        src = {}
    servers.append({"alias": alias, "url": url, "gpu": gpu, "gguf": gguf, "sha256": sha or None,
                    "quant": src.get("quant"), "source": src.get("source"), "llama_cpp": llama, "flags": flags,
                    "log": logf, "load_s": int(load_s)})
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
