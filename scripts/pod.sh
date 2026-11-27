#!/usr/bin/env bash
# Läufe auf einem Lium-Pod mit 1x RTX PRO 6000. Zwei Modi:
#
#   scripts/pod.sh test   TESTVERSION: configs/podtest.yaml, beide Familien (Qwen + Gemma, je Original +
#                         abliteriert), kurze Spiele. Die Spielzeit bis zur Deadline wird fest je zur Hälfte auf
#                         Qwen und Gemma aufgeteilt (eigener Zeitdeckel pro Gruppe). Ergebnisse + Archiv ins Log.
#   scripts/pod.sh full   HAUPTLAUF: ARENA_CONFIGS (Standard: configs/core.yaml configs/extra.yaml) ohne
#                         Zeitdeckel, fortsetzbar (Pod neu starten -> fertige Spiele werden übersprungen).
#
# Umgebung:
#   ARENA_VOLUME=/workspace   Lium-Volume (wird automatisch genommen, wenn /workspace ein Mountpoint ist):
#                             llama.cpp-Build, GGUFs und runs/ liegen dort -> nächster Pod startet in ~2 min
#   ARENA_SSH_PUBKEYS="ssh-ed25519 AAAA... du@laptop\n..."  installiert sshd und trägt die Keys ein
#   ARENA_LOG=/workspace/arena.log   Logdatei (Standard: auf dem Volume, sonst /root/arena.log)
#   test: ARENA_TEST_DEADLINE (Minuten ab Start bis Ende der Spiele, Standard 50 – passt zu --ttl 1h),
#         ARENA_TEST_CONFIG (configs/podtest.yaml), ARENA_TEST_MODELS (Standard: aus der Config)
#   full: ARENA_CONFIGS, ARENA_LIMIT (z. B. 3 = Laufzeit im Kerndesign messen)
#
# Start auf Lium siehe README ("Lauf auf Lium").
set -uo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-test}"
[[ "$MODE" == test || "$MODE" == full ]] || { echo "usage: scripts/pod.sh test|full" >&2; exit 2; }

# --- Volume -----------------------------------------------------------------------------------------
VOL="${ARENA_VOLUME:-}"
if [[ -z "$VOL" ]] && mountpoint -q /workspace 2>/dev/null; then VOL=/workspace; fi
if [[ -n "$VOL" ]]; then
  mkdir -p "$VOL/models" "$VOL/runs" "$VOL/.hf"
  export LLAMA_DIR="$VOL/llama.cpp" MODELS_DIR="$VOL/models" HF_HOME="$VOL/.hf"
  if [[ ! -L runs ]]; then rm -rf runs; ln -s "$VOL/runs" runs; fi
else
  export LLAMA_DIR=/opt/llama.cpp MODELS_DIR=/models HF_HOME=/models/.hf
fi
LOGFILE="${ARENA_LOG:-${VOL:-/root}/arena.log}"
exec > >(tee -a "$LOGFILE") 2>&1

t0=$(date +%s)
mark() { echo; echo "===== ARENA $* ($(( ($(date +%s) - t0) / 60 )) min) ====="; }
mark "START $MODE · $(date -u +%Y-%m-%dT%H:%M:%SZ) · Log: $LOGFILE · Volume: ${VOL:-keins}"

# --- SSH (optional) ---------------------------------------------------------------------------------
if [[ -n "${ARENA_SSH_PUBKEYS:-}" ]]; then
  mark "SSH"
  if ! command -v sshd >/dev/null; then
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null) \
      || echo "WARN: openssh-server konnte nicht installiert werden"
  fi
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
  printf '%b\n' "$ARENA_SSH_PUBKEYS" >>/root/.ssh/authorized_keys
  sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd && echo "sshd läuft (live zuschauen: tail -f $LOGFILE)" || echo "WARN: sshd startet nicht"
fi

# --- Setup + Modelle --------------------------------------------------------------------------------
mark "SETUP"
nvidia-smi || true
scripts/setup_gpu.sh --skip-models || { mark "FEHLER setup"; exit 1; }
source scripts/common.sh
set +e   # Pod-Skript läuft bei Fehlern einzelner Schritte weiter (Ergebnisse trotzdem ins Log)

run_test() {
  local config="${ARENA_TEST_CONFIG:-configs/podtest.yaml}"
  local models; models="${ARENA_TEST_MODELS:-$(config_models "$config" | xargs)}"
  mark "MODELLE $models"
  # shellcheck disable=SC2086
  timed "modelle bereitstellen" scripts/prepare_models.sh $models || { mark "FEHLER Modelle"; exit 1; }

  local deadline="${ARENA_TEST_DEADLINE:-50}" n_models groups left per
  n_models=$(wc -w <<<"$models"); groups=$(( (n_models + 1) / 2 ))
  # Rest bis zur Deadline minus Laden/Start-Check (~2 min pro Gruppe) und Auswertung (~2 min), fest halbiert
  left=$(( deadline - ($(date +%s) - t0) / 60 - 2 * groups - 2 ))
  per=$(( left / groups )); (( per >= 5 )) || per=5
  mark "SPIELE $config · $groups Gruppen à $per min Zeitdeckel"
  # shellcheck disable=SC2086
  ARENA_GROUP_MINUTES="$per" scripts/run_all.sh "$config" $models
  local exp; exp="$(cfg_info "$config" name)"

  mark "ZEITEN (Phase, Sekunden)"; column -t -s $'\t' "$TIMINGS" 2>/dev/null || cat "$TIMINGS"
  mark "ERGEBNISSE (results_live.jsonl)"; cat "runs/$exp/results_live.jsonl" 2>/dev/null
  mark "BENCHMARK"; cat "runs/$exp/benchmark.json" 2>/dev/null
  mark "SUMMARY"; cat "runs/$exp/summary.md" 2>/dev/null
  mark "THINKING-AUSZÜGE"
  for ev in runs/"$exp"/*/events.jsonl; do
    [[ -f "$ev" ]] || continue
    echo "--- $(basename "$(dirname "$ev")") ---"
    .venv/bin/python - "$ev" <<'PY' || true
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
nt = sum(r.get("event") == "no_tool_call" for r in rs)
rs = [r for r in rs if r.get("event") == "llm_response"]
th = [r for r in rs if r.get("reasoning")]
print(f"Antworten: {len(rs)} · mit Thinking: {len(th)} · Limit erreicht: {sum(bool(r.get('limit_hit')) for r in rs)}"
      f" · ohne Tool-Call: {nt}")
for r in th[:2]:
    t = r["reasoning"]
    print(f"[{r.get('agent')}, Aufgabe {r.get('task')}, Zug {r.get('turn')}, {len(t)} Zeichen]")
    print(t[:1500] + (" [...]" if len(t) > 1500 else ""))
PY
  done
  local archive; archive="$(ls -t runs/_exports/*.tar.gz 2>/dev/null | head -n1)"
  if [[ -n "$archive" ]]; then
    mark "ARCHIV $(basename "$archive") $(sha256sum "$archive" | cut -c1-64)"
    echo "-----BEGIN ARENA ARCHIVE-----"; base64 -w 0 "$archive"; echo; echo "-----END ARENA ARCHIVE-----"
  fi
}

run_full() {
  local configs="${ARENA_CONFIGS:-configs/core.yaml configs/extra.yaml}" c
  timed "modelle bereitstellen" scripts/prepare_models.sh || { mark "FEHLER Modelle"; exit 1; }
  for c in $configs; do
    mark "LAUF $c"
    scripts/run_all.sh "$c" || mark "FEHLER in $c (fortsetzbar: pod.sh full erneut starten)"
    mark "SUMMARY $c"; cat "runs/$(cfg_info "$c" name)/summary.md" 2>/dev/null || true
  done
  mark "ZEITEN (Phase, Sekunden)"; column -t -s $'\t' "$TIMINGS" 2>/dev/null || cat "$TIMINGS"
  echo "Ergebnisse: $(readlink -f runs) · Exporte: $(readlink -f runs)/_exports/"
}

if [[ "$MODE" == test ]]; then run_test; else run_full; fi
mark "FERTIG"
