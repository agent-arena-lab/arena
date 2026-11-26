#!/usr/bin/env bash
# Jetson AGX Orin (JetPack 6.x): gleiches Setup wie Miet-GPUs, Profil hardware/jetson.env (sm_87).
# Für stabile Zeiten vorher: sudo nvpmodel -m 0 && sudo jetson_clocks (MAXN)
ARENA_HARDWARE=jetson exec "$(dirname "$0")/setup_gpu.sh" "$@"
