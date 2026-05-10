# llama-manager

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-optimized, flicker-free curses TUI for managing a `llama.cpp` systemd service.
Zero external dependencies (uses Python stdlib only).

## Features

- **Multi-Stage Recovery:** Automatically decrements GPU layers (-ngl) on OOM/Hang, then shrinks Context Size (-c) if needed.
- **Hang Detection:** Catch silent freezes during Vulkan offloading via log stagnation checks (5m threshold).
- **Persistent Settings:** All recovery actions and settings are permanently written to the systemd service file.
- **Ultra-Smooth Scrolling:** Uses Curses Pads for instantaneous, flicker-free viewport switching.
- **Standardized Time:** All logs and status entries use **Europe/Berlin** time globally.
- **Horizontal Scroll:** Support for reading long log lines in the Journal view.
- **Clipboard Support:** Yank log lines directly to your system clipboard (requires xclip/xsel).
- **Production Ready:** Includes logging (~/.llama-manager.log) and unit tests.

## Installation

```bash
sudo ./install.sh
```
This will symlink the launcher to `/usr/local/bin/llama-manager` and optionally add the `llman` alias.

## Usage

```bash
llman
```

## Requirements

- Python 3.12+
- AMD GPU (for VRAM monitoring)
- `systemd` service named `llama.service`

## Development

```bash
# Run tests
python3 -m unittest discover -s tests -v

# Single test file
python3 -m unittest tests.test_llama_manager -v
```

## Project Structure

```
llama-manager.py      # Main curses TUI (entry point)
llama_manager.py       # Alternate/simpler variant
llama-manager.sh       # Bash launcher (sudo wrapper)
install.sh             # System-wide installation script
tests/                 # Unit tests
docs/                  # Planning docs & feature specs
legacy/                # Archived bash-based watchers
```
