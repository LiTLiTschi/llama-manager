#!/bin/bash
# llama_watch.sh — Legacy Bash wrapper
# Runs watch_llama.sh every second in a loop.
# Press 'q' to exit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_SCRIPT="$SCRIPT_DIR/watch_llama.sh"

tput civis
trap 'tput cnorm; clear; exit 0' SIGINT SIGTERM

while true; do
    bash "$WATCH_SCRIPT"
    read -rs -t 1 -n 1 k
    if [[ "$k" == "q" || "$k" == "Q" ]]; then
        tput cnorm
        clear
        exit 0
    fi
done
