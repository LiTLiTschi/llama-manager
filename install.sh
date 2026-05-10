#!/bin/bash
# install.sh - Install llama-manager to /usr/local/bin
# Usage: sudo ./install.sh
#
# Creates a symlink from /usr/local/bin/llama-manager to this repo.
# Since it's a live symlink, any git pull in the repo immediately
# updates the installed version. Run this again after git pull to
# refresh permissions/aliases.

set -e

if [ "$EUID" -ne 0 ]; then
    echo "[install] Please run as root: sudo ./install.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/llama-manager.py"
LAUNCHER="$SCRIPT_DIR/llama-manager.sh"
LINK="/usr/local/bin/llama-manager"

# Pull latest from git (non-fatal if offline or not a git repo)
echo "[install] Pulling latest changes..."
REAL_USER="${SUDO_USER:-$USER}"
if command -v git &>/dev/null && [ -d "$SCRIPT_DIR/.git" ]; then
    sudo -u "$REAL_USER" git -C "$SCRIPT_DIR" pull --ff-only 2>/dev/null || \
        echo "[install] (git pull skipped — offline or upstream unavailable)"
else
    echo "[install] (not a git repo, skipping pull)"
fi

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

# --- Symlink (update-safe) ---
if [ -L "$LINK" ]; then
    CURRENT_TARGET=$(readlink "$LINK")
    if [ "$CURRENT_TARGET" = "$LAUNCHER" ]; then
        echo "[install] Symlink already up-to-date: $LINK -> $LAUNCHER"
    else
        echo "[install] Updating symlink: $LINK -> $LAUNCHER"
        rm -f "$LINK"
        ln -s "$LAUNCHER" "$LINK"
    fi
elif [ -f "$LINK" ]; then
    echo "[install] Replacing file with symlink: $LINK -> $LAUNCHER"
    rm -f "$LINK"
    ln -s "$LAUNCHER" "$LINK"
else
    echo "[install] Creating symlink: $LINK -> $LAUNCHER"
    ln -s "$LAUNCHER" "$LINK"
fi

# --- Alias (skip on update if already present) ---
REAL_USER="${SUDO_USER:-$USER}"
HOME_DIR=$(eval echo ~$REAL_USER)
BASHRC="$HOME_DIR/.bashrc"
[ ! -f "$BASHRC" ] && BASHRC="/root/.bashrc"

ALIAS_LINE="alias llman='llama-manager'"

if grep -qF "$ALIAS_LINE" "$BASHRC" 2>/dev/null; then
    echo "[install] Alias 'llman' already exists in $BASHRC."
else
    # Only prompt on fresh install, not update
    if [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$LAUNCHER" ]; then
        # Symlink was already correct before we touched it — this is an update
        printf '\n# llama-manager alias\n%s\n' "$ALIAS_LINE" >> "$BASHRC"
        echo "[install] Alias 'llman' added to $BASHRC."
    else
        printf "\n[install] Add alias 'llman=llama-manager' to %s? [Y/n] " "$BASHRC"
        read -r answer < /dev/tty || answer="Y"
        answer=${answer:-Y}
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            printf '\n# llama-manager alias\n%s\n' "$ALIAS_LINE" >> "$BASHRC"
            echo "[install] Alias added."
        fi
    fi
fi

echo "[install] Done. Run 'llman' or 'llama-manager'."
