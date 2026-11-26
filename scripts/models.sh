#!/usr/bin/env bash
# Model catalog: list, speed-test and choose the models for the experiment.
#
#   scripts/models.sh list                    Katalog + letzte Messwerte
#   scripts/models.sh bench [-y] [model ...]  Laden (falls nötig), Server starten, Geschwindigkeit messen
#                                             ohne Modellnamen: Auswahlmenü
#   scripts/models.sh choose [model ...]      Modelle für run_all.sh / prepare_models.sh festlegen
#                                             (schreibt ARENA_MODELS in .env); ohne Namen: Auswahlmenü
#
# Ergebnisse: runs/_bench/results.jsonl (eine Zeile pro Messung) und runs/_bench/latest.md
source "$(dirname "$0")/common.sh"

main() {
BENCH_DIR="$ROOT_DIR/runs/_bench"
RESULTS="$BENCH_DIR/results.jsonl"
mkdir -p "$BENCH_DIR"
PY="$ROOT_DIR/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || die "python3 not found"

# Interactive input also works with `curl ... | bash` (stdin is the script then)
ask() { local prompt="$1" ans=""; if [[ -r /dev/tty ]]; then read -r -p "$prompt" ans </dev/tty || true; fi; printf '%s' "$ans"; }

ram_gb() {
  if [[ -r /proc/meminfo ]]; then awk '/MemTotal/{printf "%d", $2/1024/1024 + 0.5}' /proc/meminfo
  else echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 )); fi
}
RAM="$(ram_gb)"

catalog() {  # one line per model: name|type|size|min_ram|downloaded
  local m
  while IFS= read -r m; do
    load_model_env "$m"
    printf '%s|%s|%s|%s|%s\n' "$m" "${TYPE:-?}" "${SIZE_GB:-?}" "${MIN_RAM_GB:-0}" "$([[ -n "$MODEL_PATH" ]] && echo ja || echo nein)"
  done < <(all_models)
}

table() {  # numbered table of the catalog with the latest measurement per model (on this host)
  catalog | "$PY" -c '
import json, socket, sys
res = {}
try:
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r.get("host") == socket.gethostname():
            res[r["model"]] = r
except FileNotFoundError:
    pass
chosen = set(sys.argv[3].split())
ram = int(sys.argv[2] or 0)
print(f"RAM dieses Geräts: {ram} GB   (* = aktuell ausgewählt)\n")
hdr = ("#", "Modell", "Typ", "GB", "geladen", "pp tok/s", "tg tok/s", "Tool-Call", "GPU", "min/Spiel*", "h/200 Spiele*")
rows = []
for i, line in enumerate(sys.stdin, 1):
    name, typ, size, minram, dl = line.strip().split("|")
    r = res.get(name, {})
    fits = ram == 0 or int(minram) <= ram
    gpu = r.get("backend", "")
    if r.get("gpu_layers"):
        gpu += " " + r["gpu_layers"]
    rows.append((
        str(i), ("*" if name in chosen else " ") + name, typ, size, dl if fits else f"{dl} (zu wenig RAM)",
        str(r.get("pp_tok_s", "")), str(r.get("tg_tok_s", "")),
        "" if not r else ("native" if r.get("tool_call_native") else "FEHLT"),
        gpu, str(r.get("est_min_per_run", "")), str(r.get("est_h_per_200_games", "")),
    ) if not r.get("error") else (str(i), name, typ, size, dl, "Fehler: " + r["error"][:60], "", "", "", "", ""))
w = [max(len(x) for x in col) for col in zip(hdr, *rows)]
fmt = "  ".join("{:<%d}" % n for n in w)
print(fmt.format(*hdr)); print(fmt.format(*["-" * n for n in w]))
for r in rows:
    print(fmt.format(*r))
print("\n* grobe Schätzung (Notion): Zeit pro Spiel ~ 30.000 / tg tok/s, ein Spiel nach dem anderen auf einer GPU.")
print("  Parallele Spiele (GAMES_PER_SERVER) erhöhen den Durchsatz etwa um Faktor 2 (2 Spiele) bzw. 3-3,5 (4 Spiele).")
' "$RESULTS" "$RAM" "${ARENA_MODELS:-}"
}

pick() {  # $1 = question; prints the chosen model names
  local names=() m ans i
  while IFS= read -r m; do names+=("$m"); done < <(all_models)
  ans="$(ask "$1")"
  for i in $ans; do
    if [[ "$i" =~ ^[0-9]+$ ]] && (( i >= 1 && i <= ${#names[@]} )); then echo "${names[$((i-1))]}"
    elif [[ -f "$ROOT_DIR/models/$i.env" ]]; then echo "$i"
    else log "ignoriere '$i'"; fi
  done
}

bench_one() {
  local m="$1"
  load_model_env "$m"
  log "===== Speedtest: $m ($TYPE, ${SIZE_GB} GB) ====="
  # Speedtest: genau ein Server auf der ersten GPU, sonst gleiche Einstellungen wie im Experiment
  local first_gpu; first_gpu="$(detect_hardware; cut -d, -f1 <<<"$GPU_IDS")"
  CUDA_VISIBLE_DEVICES="$first_gpu" MODELS_PER_GPU=1 "$ROOT_DIR/scripts/serve.sh" start "$m"
  local rc=0 logf ctx
  logf="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["servers"][0]["log"])' "$RUNS_DIR/_system/servers.json")"
  ctx="$(grep -oE -- '-c [0-9]+' <<<"$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["servers"][0]["flags"])' "$RUNS_DIR/_system/servers.json")" | awk '{print $2}')"
  "$PY" "$ROOT_DIR/scripts/bench_model.py" --base-url "http://$LLM_HOST:$LLM_BASE_PORT" --alias "$ALIAS" \
    --type "$TYPE" --size-gb "$SIZE_GB" --ctx "$ctx" --server-log "$logf" --out "$RESULTS" >/dev/null || rc=$?
  "$ROOT_DIR/scripts/serve.sh" stop
  [[ $rc -eq 0 ]] || log "$m: Messung mit Fehler (steht in der Tabelle)"
}

cmd_bench() {
  local yes=0 models=() m todo=() gb=0
  [[ "${1:-}" == "-y" ]] && { yes=1; shift; }
  models=("$@")
  if [[ ${#models[@]} -eq 0 ]]; then
    table
    echo
    while IFS= read -r m; do models+=("$m"); done < <(pick "Welche Modelle testen? Nummern mit Leerzeichen (z. B. 1 3 4): ")
  fi
  [[ ${#models[@]} -gt 0 ]] || die "keine Modelle gewählt"
  for m in "${models[@]}"; do
    load_model_env "$m"
    if (( RAM > 0 && ${MIN_RAM_GB:-0} > RAM )); then log "$m: braucht ~${MIN_RAM_GB} GB RAM, Gerät hat $RAM GB -> übersprungen"; continue; fi
    todo+=("$m")
    [[ -n "$MODEL_PATH" ]] || gb="$("$PY" -c "print(round($gb + ${SIZE_GB:-0}, 1))")"
  done
  [[ ${#todo[@]} -gt 0 ]] || die "nichts zu testen"
  if [[ "$gb" != "0" && $yes -eq 0 ]]; then
    [[ "$(ask "Download von ca. $gb GB nötig. Fortfahren? [j/N] ")" =~ ^[jJyY] ]] || die "abgebrochen"
  fi
  "$ROOT_DIR/scripts/prepare_models.sh" "${todo[@]}"
  trap '"$ROOT_DIR/scripts/serve.sh" stop' EXIT INT TERM
  for m in "${todo[@]}"; do bench_one "$m"; done
  table | tee "$BENCH_DIR/latest.md"
  echo
  log "Gespeichert: runs/_bench/results.jsonl, runs/_bench/latest.md"
  log "Auswählen: scripts/models.sh choose"
}

cmd_choose() {
  local models=("$@") m
  if [[ ${#models[@]} -eq 0 ]]; then
    table
    echo
    while IFS= read -r m; do models+=("$m"); done < <(pick "Welche Modelle sollen laufen? Nummern mit Leerzeichen: ")
  fi
  [[ ${#models[@]} -gt 0 ]] || die "keine Modelle gewählt"
  for m in "${models[@]}"; do [[ -f "$ROOT_DIR/models/$m.env" ]] || die "unbekanntes Modell '$m'"; done
  local env="$ROOT_DIR/.env" line="ARENA_MODELS=\"${models[*]}\""
  touch "$env"
  grep -v '^ARENA_MODELS=' "$env" >"$env.tmp" || true
  echo "$line" >>"$env.tmp" && mv "$env.tmp" "$env"
  log "ausgewählt: ${models[*]}  (gespeichert in .env)"
  log "run_all.sh und prepare_models.sh nehmen jetzt diese Modelle, wenn kein Modell angegeben ist."
  log "Modelle, die nicht unter models: in der Config stehen, bekommen ihre Einstellungen aus configs/model_catalog.yaml."
}

case "${1:-list}" in
  list) table ;;
  bench) shift; cmd_bench "$@" ;;
  choose) shift; cmd_choose "$@" ;;
  *) die "usage: models.sh list | bench [-y] [model ...] | choose [model ...]" ;;
esac
}

main "$@"
