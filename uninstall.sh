#!/usr/bin/env bash
# Entfernt ALLES, was install.sh angelegt hat – inklusive runs/ mit allen Versuchsdaten.
# Es gibt KEIN Backup. Vorher sichern, was behalten werden soll.
#
# Einzeiler:
#   gh api repos/agent-arena-lab/arena/contents/uninstall.sh -H "Accept: application/vnd.github.raw" | bash
# Ohne Rückfrage:
#   gh api repos/agent-arena-lab/arena/contents/uninstall.sh -H "Accept: application/vnd.github.raw" | bash -s -- --yes
#
# Gelöscht wird nur, was in ~/.arena/manifest steht (also von install.sh angelegt wurde):
#   Installationsordner (Repo, runs/, Modelle, HF-Cache, venv, llama.cpp), Docker-Images,
#   brew-/apt-Pakete, die install.sh selbst installiert hat, und zuletzt ~/.arena.
set -euo pipefail

YES=0
for a in "$@"; do
  case "$a" in
    -y|--yes) YES=1 ;;
    *) echo "Unbekannte Option: $a" >&2; exit 1 ;;
  esac
done

ARENA_HOME="$HOME/.arena"
MANIFEST="$ARENA_HOME/manifest"
[[ -f "$MANIFEST" ]] || { echo "Keine arena-Installation gefunden ($MANIFEST fehlt). Nichts zu tun."; exit 0; }

DIRS=(); BREW=(); APT=(); IMAGES=()
while IFS=$'\t' read -r type value; do
  [[ -n "${type:-}" && -n "${value:-}" ]] || continue
  case "$type" in
    dir) DIRS+=("$value") ;;
    brew) BREW+=("$value") ;;
    apt) APT+=("$value") ;;
    docker-image) IMAGES+=("$value") ;;
  esac
done <"$MANIFEST"

echo "Folgendes wird ENDGÜLTIG gelöscht:"
for d in ${DIRS[@]+"${DIRS[@]}"}; do
  size="$(du -sh "$d" 2>/dev/null | cut -f1 || true)"
  echo "  Ordner:        $d ${size:+($size, inkl. runs/ und Modelle)}"
done
for i in ${IMAGES[@]+"${IMAGES[@]}"}; do echo "  Docker-Image:  $i (+ Container daraus)"; done
for b in ${BREW[@]+"${BREW[@]}"}; do echo "  brew-Paket:    $b"; done
for p in ${APT[@]+"${APT[@]}"}; do echo "  apt-Paket:     $p"; done
echo "  Protokoll:     $ARENA_HOME"
echo

if [[ $YES -eq 0 ]]; then
  if [[ ! -r /dev/tty ]]; then
    echo "Keine Rückfrage möglich (kein Terminal). Mit --yes erneut starten." >&2
    exit 1
  fi
  printf "Wirklich ALLES löschen? Es gibt kein Backup. Zum Bestätigen 'ja' eingeben: "
  read -r answer </dev/tty
  [[ "$answer" == "ja" ]] || { echo "Abgebrochen. Nichts gelöscht."; exit 1; }
fi

cd "$HOME"   # falls das Skript aus dem Installationsordner heraus gestartet wurde

# 1. laufenden llama-server stoppen
for d in ${DIRS[@]+"${DIRS[@]}"}; do
  pid_file="$d/.state/llama-server.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null && echo "llama-server gestoppt (pid $pid)" || true
  fi
done

# 2. Docker-Container und -Images
if [[ ${#IMAGES[@]} -gt 0 ]] && command -v docker >/dev/null; then
  for i in "${IMAGES[@]}"; do
    ids="$(docker ps -aq --filter "ancestor=$i" 2>/dev/null || true)"
    [[ -n "$ids" ]] && docker rm -f $ids >/dev/null && echo "Container entfernt ($i)"
    docker image inspect "$i" >/dev/null 2>&1 && docker rmi -f "$i" >/dev/null && echo "Image entfernt: $i"
  done
fi

# 3. Installationsordner (nur mit Sicherheitsmarker, nie $HOME oder /)
for d in ${DIRS[@]+"${DIRS[@]}"}; do
  if [[ -z "$d" || "$d" == "/" || "$d" == "$HOME" ]]; then
    echo "Überspringe unsicheren Pfad: '$d'" >&2; continue
  fi
  if [[ ! -e "$d" ]]; then echo "Bereits weg: $d"; continue; fi
  if [[ ! -f "$d/.arena-installed" ]]; then
    echo "Überspringe $d (kein .arena-installed-Marker -> nicht von install.sh angelegt)" >&2; continue
  fi
  rm -rf "$d" 2>/dev/null || sudo rm -rf "$d"
  echo "Ordner gelöscht: $d"
done

# 4. Pakete, die install.sh selbst installiert hat
for b in ${BREW[@]+"${BREW[@]}"}; do
  brew uninstall "$b" >/dev/null 2>&1 && echo "brew-Paket entfernt: $b" || echo "brew-Paket nicht entfernt: $b" >&2
done
if [[ ${#APT[@]} -gt 0 ]]; then
  sudo apt-get remove -y -qq "${APT[@]}" && echo "apt-Pakete entfernt: ${APT[*]}"
fi

# 5. Protokoll zuletzt
rm -rf "$ARENA_HOME"
echo "Fertig. arena ist vollständig entfernt."
