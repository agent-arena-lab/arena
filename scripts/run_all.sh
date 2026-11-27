#!/usr/bin/env bash
# Komplette Pipeline auf der RTX PRO 6000: pro Familie (Original + abliteriert gleichzeitig auf der GPU)
# llama-server starten -> Start-Check -> Spiele -> (Manipulationscheck) -> Server stoppen; am Ende Auswertung
# und Export. Fortsetzbar: fertige Spiele (result.json) werden übersprungen.
#
#   scripts/run_all.sh [config] [model ...]
#   scripts/run_all.sh configs/podtest.yaml               Testversion (kurz, alle vier Modelle)
#   scripts/run_all.sh configs/pilot.yaml                 Pilot ohne Thinking-Limit, info_hint an/aus
#   scripts/run_all.sh configs/core.yaml                  Kerndesign, 160 Spiele
#   scripts/run_all.sh configs/extra.yaml                 feste Zusatzläufe, 40 Spiele
#   ARENA_LIMIT=3 scripts/run_all.sh configs/core.yaml    nur 3 Spiele pro Gruppe (Laufzeit messen)
#   ARENA_GROUP_MINUTES=20 ...                            Zeitdeckel pro Familie: danach keine neuen Spiele,
#                                                         harter Abbruch nach weiteren 25 % (mind. 5 min)
#   ARENA_MANIPCHECK=0 ...                                ohne Manipulationscheck
source "$(dirname "$0")/common.sh"

main() {
cd "$ROOT_DIR"
CONFIG="${1:-configs/core.yaml}"; shift || true
[[ -f "$ROOT_DIR/$CONFIG" ]] || die "config $CONFIG not found"
models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/$CONFIG"); fi
[[ ${#models[@]} -gt 0 ]] || die "keine Modelle in $CONFIG"
EXP_NAME="$(cfg_info "$ROOT_DIR/$CONFIG" name)"
SYS_DIR="$RUNS_DIR/$EXP_NAME/_system"
mkdir -p "$SYS_DIR"

detect_hardware
export REASONING_BUDGET REASONING_BUDGET_MESSAGE
REASONING_BUDGET="$(cfg_info "$ROOT_DIR/$CONFIG" thinking.budget)"
REASONING_BUDGET_MESSAGE="$(cfg_info "$ROOT_DIR/$CONFIG" thinking.budget_message)"
CTX_CFG="$(cfg_info "$ROOT_DIR/$CONFIG" thinking.ctx_per_slot)"
if (( CTX_CFG > CTX_PER_SLOT )); then
  # längerer Kontext (Pilot ohne Limit): gleicher KV-Speicher, dafür weniger Spiele gleichzeitig
  kv=$(( GAMES_PER_SERVER * CTX_PER_SLOT ))
  GAMES_PER_SERVER=$(( kv / CTX_CFG )); (( GAMES_PER_SERVER >= 1 )) || GAMES_PER_SERVER=1
  CTX_PER_SLOT="$CTX_CFG"
  export GAMES_PER_SERVER CTX_PER_SLOT
fi
log "$(hardware_summary)"
log "Config $CONFIG · Thinking-Budget $REASONING_BUDGET · Budget-Meldung: \"$REASONING_BUDGET_MESSAGE\""

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host: $(hostname)"; uname -a
  hardware_summary
  nvidia-smi 2>/dev/null || true
  "$LLAMA_SERVER" --version 2>&1 | head -n 5 || true
  git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/llama.cpp commit: /' || true
  free -h 2>/dev/null || true
} >"$SYS_DIR/system_$(date +%Y%m%d-%H%M%S).txt" 2>&1

trap '"$ROOT_DIR/scripts/serve.sh" stop' EXIT INT TERM
slots="$MODELS_PER_GPU"
run_args=(); [[ -n "${ARENA_LIMIT:-}" ]] && run_args+=(--limit "$ARENA_LIMIT")
[[ "${ARENA_QUIET:-0}" == "1" ]] && run_args+=(--quiet)
n_groups=$(( (${#models[@]} + slots - 1) / slots ))
g=0

for (( start = 0; start < ${#models[@]}; start += slots )); do
  g=$(( g + 1 ))
  group=("${models[@]:start:slots}")
  log "===== Gruppe $g/$n_groups: ${group[*]} ====="
  timed "$EXP_NAME: server starten ${group[*]}" "$ROOT_DIR/scripts/serve.sh" start "${group[@]}"
  cp "$RUNS_DIR/_system/servers.json" "$SYS_DIR/servers_$(date +%Y%m%d-%H%M%S).json"
  timed "$EXP_NAME: start-check ${group[*]}" harness benchmark --config "$CONFIG" --model "${group[@]}" \
    || die "Start-Check fehlgeschlagen (runs/$EXP_NAME/benchmark.json)"
  cp "$RUNS_DIR/$EXP_NAME/benchmark.json" "$SYS_DIR/benchmark_$(date +%Y%m%d-%H%M%S).json"
  args=("${run_args[@]+"${run_args[@]}"}")
  if [[ -n "${ARENA_GROUP_MINUTES:-}" ]]; then
    hard=$(( ARENA_GROUP_MINUTES + (ARENA_GROUP_MINUTES / 4 > 5 ? ARENA_GROUP_MINUTES / 4 : 5) ))
    log "Zeitdeckel Gruppe $g: $ARENA_GROUP_MINUTES min (keine neuen Spiele danach), harter Abbruch nach $hard min"
    rc=0
    timed "$EXP_NAME: spiele ${group[*]}" timeout --signal=INT --kill-after=60 "$(( hard * 60 ))" \
      "$ROOT_DIR/.venv/bin/arena" run --config "$CONFIG" --model "${group[@]}" \
      --games-per-server "$GAMES_PER_SERVER" --time-budget-min "$ARENA_GROUP_MINUTES" ${args[@]+"${args[@]}"} || rc=$?
    [[ $rc -eq 124 || $rc -eq 130 ]] && log "Gruppe $g hart abgebrochen (laufende Spiele bleiben unvollständig, werden beim nächsten Start neu gespielt)"
  else
    timed "$EXP_NAME: spiele ${group[*]}" harness run --config "$CONFIG" --model "${group[@]}" \
      --games-per-server "$GAMES_PER_SERVER" ${args[@]+"${args[@]}"} || log "Spiele mit Fehler beendet (fortsetzbar)"
  fi
  if [[ "${ARENA_MANIPCHECK:-1}" == "1" && "$EXP_NAME" == core ]]; then
    timed "$EXP_NAME: manipulationscheck ${group[*]}" harness manipcheck --config "$CONFIG" --model "${group[@]}" >/dev/null \
      && log "Manipulationscheck: runs/$EXP_NAME/_manipulation_check/"
  fi
  "$ROOT_DIR/scripts/serve.sh" stop
done

timed "$EXP_NAME: auswertung" harness analyze "runs/$EXP_NAME" || true
timed "$EXP_NAME: export" harness export "runs/$EXP_NAME" || true
log "fertig: runs/$EXP_NAME/{results_live.jsonl,results.csv,summary.md,benchmark.json} · Zeiten: $TIMINGS"
}

main "$@"
