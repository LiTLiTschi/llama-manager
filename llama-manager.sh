#!/bin/bash
# llama-manager.sh - Entrypoint for the llama.cpp curses TUI manager.
# Resolves symlinks so SCRIPT_DIR always points to the real repo location.

if [ "$EUID" -ne 0 ]; then
  exec sudo "$0" "$@"
fi

# readlink -f resolves the symlink to the actual file path
REAL_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$REAL_PATH")"

exec python3 "$SCRIPT_DIR/llama-manager.py"
