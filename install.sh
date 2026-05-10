#!/bin/bash
# install.sh - Install llama-manager to /usr/local/bin
# Usage: sudo ./install.sh

set -e

if [ "$EUID" -ne 0 ]; then
    echo "[install] Please run as root: sudo ./install.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/llama-manager.py"
LAUNCHER="$SCRIPT_DIR/llama-manager.sh"
LINK="/usr/local/bin/llama-manager"

echo "[install] Checking dependencies..."
for cmd in python3 journalctl systemctl; do
    if ! command -v $cmd &>/dev/null; then
        echo "[install] Warning: $cmd not found. This tool requires $cmd."
    fi
done

if ! command -v xclip &>/dev/null && ! command -v xsel &>/dev/null; then
    echo "[install] Note: 'xclip' or 'xsel' not found. Yanking to clipboard will be disabled."
fi

echo "[install] Setting permissions..."
chmod +x "$PYTHON_SCRIPT"
chmod +x "$LAUNCHER"

if [ -L "$LINK" ] || [ -f "$LINK" ]; then
    rm -f "$LINK"
fi

ln -s "$LAUNCHER" "$LINK"
echo "[install] Symlinked $LAUNCHER to $LINK"

# --- Alias prompt ---
REAL_USER="${SUDO_USER:-$USER}"
HOME_DIR=$(eval echo ~$REAL_USER)
BASHRC="$HOME_DIR/.bashrc"
[ ! -f "$BASHRC" ] && BASHRC="/root/.bashrc"

ALIAS_LINE="alias llman='llama-manager'"

if grep -qF "$ALIAS_LINE" "$BASHRC" 2>/dev/null; then
    echo "[install] Alias 'llman' already exists in $BASHRC."
else
    printf "\n[install] Add alias 'llman=llama-manager' to %s? [Y/n] " "$BASHRC"
    # Try to read from /dev/tty to handle being run via sudo/pipes
    read -r answer < /dev/tty || answer="Y"
    answer=${answer:-Y}
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        printf '\n# llama-manager alias\n%s\n' "$ALIAS_LINE" >> "$BASHRC"
        echo "[install] Alias added."
    fi
fi

echo "[install] Installation complete. Run 'llman' or 'llama-manager'."
