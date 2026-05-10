#!/usr/bin/env python3
"""
llama-manager.py - curses-based TUI for managing llama.cpp systemd service.
Production-ready with Terminal Failure detection and flicker-free pads.
"""

import curses
import os
import re
import subprocess
import time
import glob
import threading
import logging
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

SERVICE_NAME = "llama.service"
SERVICE_FILE = f"/etc/systemd/system/{SERVICE_NAME}"
LOG_FILE = Path.home() / ".llama-manager.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_REAL_USER = os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
_REAL_HOME = Path(f"/home/{_REAL_USER}") if _REAL_USER != "root" else Path("/root")
GGUF_DIR = _REAL_HOME / ".gguf"

MENU = [
    ("Watch Mode", "watch"),
    ("Journal", "logs"),
    ("Journal (Pager)", "journal"),
    ("Restart & Timeout Settings", "auto_restart_settings"),
    ("Start", "start"),
    ("Stop", "stop"),
    ("Restart", "restart"),
    ("Reset Error", "reset_error"),
    ("Quit", "quit"),
]


@dataclass
class Setting:
    key: str
    label: str
    flag: str
    type: str  # "int", "bool", "enum", "string"
    default: Any = None
    options: list = field(default_factory=list)
    description: str = ""


# Auto-restart settings registry - each entry becomes a submenu item
SETTINGS: List[Setting] = [
    Setting(
        "ngl_start",
        "GPU Layers (-ngl)",
        "-ngl",
        "int",
        0,
        description="Number of GPU layers to offload",
    ),
    Setting(
        "ngl_step",
        "NGL Decrement Per Retry",
        "",
        "int",
        1,
        description="How much to reduce -ngl each retry",
    ),
    Setting(
        "retry_on_oom",
        "Retry on OOM Crash",
        "",
        "bool",
        False,
        description="Auto-retry with lower -ngl after OOM",
    ),
    Setting(
        "ctx_start",
        "Context Size (-c)",
        "-c",
        "int",
        2048,
        description="Context window size in tokens",
    ),
    Setting(
        "ctx_step",
        "Context Decrement Per Retry",
        "",
        "int",
        2048,
        description="How much to reduce -c each retry",
    ),
    Setting(
        "enable_ctx_reduction",
        "Enable Context Reduction",
        "",
        "bool",
        True,
        description="Reduce context when ngl reaches 0",
    ),
    Setting(
        "ctx_reduction_pct",
        "Context Reduction %",
        "",
        "int",
        12,
        description="Percentage to reduce context per retry",
    ),
    Setting(
        "hang_timeout_mins",
        "Hang Detection Timeout (min)",
        "",
        "int",
        15,
        description="Minutes without response before declaring hang",
    ),
    Setting(
        "stagnation_timeout_mins",
        "Log Stagnation Timeout (min)",
        "",
        "int",
        5,
        description="Minutes without log output before declaring stuck",
    ),
    Setting(
        "oom_detection_timeout_secs",
        "OOM Detection Timeout (sec)",
        "",
        "int",
        30,
        description="Seconds after restart to detect OOM crash",
    ),
    Setting(
        "auto_recovery_enabled",
        "Auto-Recovery Enabled",
        "",
        "bool",
        True,
        description="Master toggle for automated recovery",
    ),
    Setting(
        "host",
        "Host Address",
        "--host",
        "string",
        "127.0.0.1",
        description="Network interface to bind to",
    ),
    Setting("port", "Port", "--port", "int", 8080, description="Server listening port"),
    Setting(
        "server_timeout",
        "Server Timeout (sec)",
        "--timeout",
        "int",
        600,
        description="HTTP server read timeout",
    ),
    Setting(
        "threads",
        "CPU Threads (-t)",
        "-t",
        "int",
        0,
        options=[],
        description="Number of CPU threads (0=auto)",
    ),
    Setting(
        "threads_batch",
        "Batch Threads (-tb)",
        "-tb",
        "int",
        0,
        description="Number of batch processing threads (0=auto)",
    ),
    Setting(
        "batch_size",
        "Batch Size (-b)",
        "-b",
        "int",
        2048,
        description="Prompt processing batch size",
    ),
    Setting(
        "ubatch_size",
        "Micro Batch Size (-ub)",
        "-ub",
        "int",
        512,
        description="Physical batch size for token generation",
    ),
    Setting(
        "mlock",
        "mlock (Lock RAM)",
        "--mlock",
        "bool",
        False,
        description="Lock model in RAM against swapping",
    ),
    Setting(
        "flash_attn",
        "Flash Attention",
        "--flash-attn",
        "enum",
        "auto",
        options=["on", "off", "auto", "1", "0"],
        description="Use flash attention",
    ),
    Setting(
        "cache_type_k",
        "K Cache Type",
        "-ctk",
        "enum",
        "f16",
        options=[
            "f32",
            "f16",
            "bf16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
        ],
        description="Key cache quantization type",
    ),
    Setting(
        "cache_type_v",
        "V Cache Type",
        "-ctv",
        "enum",
        "f16",
        options=[
            "f32",
            "f16",
            "bf16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
        ],
        description="Value cache quantization type",
    ),
    Setting(
        "cache_ram",
        "CPU Cache RAM (MB)",
        "--cache-ram",
        "int",
        8192,
        description="CPU cache RAM limit in MB",
    ),
    Setting(
        "parallel_slots",
        "Parallel Slots (-np)",
        "-np",
        "int",
        0,
        description="Max parallel request slots (0=auto)",
    ),
    Setting(
        "model_path",
        "Model Path (-m)",
        "-m",
        "string",
        "",
        description="Path to GGUF model file",
    ),
    Setting(
        "no_kv_offload",
        "No KV Offload",
        "--no-kv-offload",
        "bool",
        False,
        description="Disable KV cache offload to GPU",
    ),
    Setting(
        "reasoning_budget",
        "Reasoning Budget",
        "--reasoning-budget",
        "int",
        -1,
        description="Token budget for thinking (-1=unrestricted)",
    ),
    Setting(
        "spec_type",
        "Speculative Type",
        "--spec-type",
        "enum",
        "none",
        options=[
            "none",
            "mtp",
            "ngram-cache",
            "ngram-simple",
            "ngram-map-k",
            "ngram-map-k4v",
            "ngram-mod",
        ],
        description="Speculative decoding type",
    ),
    Setting(
        "no_mmap",
        "No Memory Map",
        "--no-mmap",
        "bool",
        False,
        description="Disable memory-mapped model loading",
    ),
    Setting(
        "alias",
        "Model Alias",
        "-a",
        "string",
        "",
        description="Model name alias for API",
    ),
    Setting(
        "watch_log_lines",
        "Watch Log Lines",
        "",
        "int",
        6,
        description="Number of recent log entries shown in watch mode",
    ),
]


_JOURNAL_PREFIX = re.compile(r"^\w{3}\s+\d+\s+[\d:]+\s+\S+\s+\S+\[\d+\]:\s*")
_LLAMA_MANAGER_META_PREFIX = "# llama-manager:"
_OOM_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "oom",
    "cannot meet free memory target",
    "failed to fit params",
)

# ---------------------------------------------------------------------------
# Global State & Workers
# ---------------------------------------------------------------------------


class GlobalState:
    def __init__(self):
        self.service_status = "STOPPED"
        self.ram, self.vram, self.prog = "N/A", "N/A", "Idle"
        self.meta, self.ngl, self.ctx, self.slots = {}, 0, 0, {}
        self.is_ready = False
        self.loading_since: Optional[float] = None
        self.last_log_ts: Optional[float] = None
        self.last_log_msg = ""
        self.oom_seen = False
        self.critical_error: Optional[str] = None
        self.last_manual_reset = 0.0
        self.wrap_mode = True
        self.lock = threading.Lock()

    def refresh(self):
        s = (
            "RUNNING"
            if _run(["systemctl", "is-active", "--quiet", SERVICE_NAME]).returncode == 0
            else "STOPPED"
        )
        r, v = get_ram(), get_vram()
        try:
            t = Path(SERVICE_FILE).read_text()
            m, n, c = (
                _read_all_meta(t),
                _extract_execstart_ngl(t),
                _extract_execstart_ctx(t),
            )
        except Exception:
            m, n, c = {}, 0, 0

        with self.lock:
            self.service_status, self.ram, self.vram, self.meta, self.ngl, self.ctx = (
                s,
                r,
                v,
                m,
                n,
                c,
            )
            if self.meta.get("auto_recovery_enabled", "true") == "false":
                return
            if self.critical_error:
                return
            if time.time() - self.last_manual_reset < 5:
                return

            if self.oom_seen:
                self._trigger_recovery("OOM Crash")
                self.oom_seen = False
                return

            if s == "RUNNING" and not self.is_ready:
                now = time.time()
                if self.loading_since is None:
                    self.loading_since = now
                if (now - self.loading_since) / 60 >= int(
                    self.meta.get("hang_timeout_mins", 15)
                ):
                    self._trigger_recovery("Loading Timeout")
                    return
                if self.last_log_ts and (now - self.last_log_ts) / 60 >= int(
                    self.meta.get("stagnation_timeout_mins", 5)
                ):
                    self._trigger_recovery(
                        f"Log Stagnation (Stuck at: {self.last_log_msg[:30]})"
                    )
            else:
                self.loading_since = None

    def _trigger_recovery(self, reason: str):
        if self.ngl == 0 and self.meta.get("enable_ctx_reduction", "true") == "false":
            self.critical_error = (
                f"CRITICAL FAILURE: {reason}. Manual Intervention Required."
            )
            logging.warning(
                f"Recovery blocked: {reason}, ngl=0, ctx reduction disabled"
            )
            return

        updates = {"last_recovery_reason": reason}
        if "OOM" in reason:
            updates["oom_restart_count"] = (
                int(self.meta.get("oom_restart_count", 0)) + 1
            )
        else:
            updates["hang_recovery_count"] = (
                int(self.meta.get("hang_recovery_count", 0)) + 1
            )

        ngl_step = int(self.meta.get("ngl_decrement_step", 1))
        if self.ngl > 0:
            updates["ngl_start"] = max(0, self.ngl - ngl_step)
            logging.info(
                f"Recovery: {reason}, decrementing ngl {self.ngl} -> {updates['ngl_start']}"
            )
        else:
            reduction_pct = float(self.meta.get("ctx_reduction_pct", 12.5)) / 100.0
            new_ctx = max(2048, self.ctx - max(2048, int(self.ctx * reduction_pct)))
            if new_ctx != self.ctx:
                updates["ctx_start"] = new_ctx
                logging.info(
                    f"Recovery: {reason}, reducing ctx {self.ctx} -> {new_ctx}"
                )
            else:
                self.critical_error = (
                    f"CRITICAL FAILURE: {reason}. All automated recovery exhausted."
                )
                logging.error(
                    f"Recovery exhausted: {reason}, ngl=0, ctx stuck at {self.ctx}"
                )

        _write_all_meta(updates)
        daemon_reload_if_needed()
        _run(["systemctl", "restart", SERVICE_NAME])
        self.loading_since = self.last_log_ts = None


gs = GlobalState()


class LogManager:
    def __init__(self, max_lines=5000):
        self.raw_logs, self.cursor = deque(maxlen=max_lines), None
        self.lock, self.stop_event, self.new_data_event = (
            threading.Lock(),
            threading.Event(),
            threading.Event(),
        )

    def start(self):
        self.new_data_event.set()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        sc = self._get_last_start_cursor()
        cmd = (
            [
                "journalctl",
                "-u",
                SERVICE_NAME,
                "--after-cursor",
                sc,
                "-o",
                "short-iso",
                "--show-cursor",
                "--no-pager",
            ]
            if sc
            else [
                "journalctl",
                "-u",
                SERVICE_NAME,
                "-n",
                "500",
                "-o",
                "short-iso",
                "--show-cursor",
                "--no-pager",
            ]
        )
        self._fetch(cmd)
        while not self.stop_event.is_set():
            cmd = (
                [
                    "journalctl",
                    "-u",
                    SERVICE_NAME,
                    "--after-cursor",
                    self.cursor,
                    "-o",
                    "short-iso",
                    "--show-cursor",
                    "--no-pager",
                ]
                if self.cursor
                else [
                    "journalctl",
                    "-u",
                    SERVICE_NAME,
                    "-n",
                    "100",
                    "-o",
                    "short-iso",
                    "--show-cursor",
                    "--no-pager",
                ]
            )
            if self._fetch(cmd):
                self.new_data_event.set()
                self._update_gs()
            self.stop_event.wait(1.5)

    def _get_last_start_cursor(self) -> Optional[str]:
        try:
            out = _run(
                [
                    "journalctl",
                    "-u",
                    SERVICE_NAME,
                    "MESSAGE=Started llama.service.",
                    "-n",
                    "1",
                    "-o",
                    "short-iso",
                    "--show-cursor",
                    "--no-pager",
                ],
                capture_output=True,
                text=True,
            ).stdout
            for l in out.splitlines():
                if l.startswith("-- cursor:"):
                    return l.split(": ")[1].strip()
        except Exception:
            pass
        return None

    def _fetch(self, cmd: List[str]) -> bool:
        try:
            out = _run(cmd, capture_output=True, text=True, timeout=5).stdout
            added, last_line, oom_found = False, "", False
            for l in out.strip().splitlines():
                if l.startswith("-- cursor:"):
                    self.cursor = l.split(": ")[1].strip()
                elif l.strip() and not l.startswith("-- "):
                    clean = strip_journal_prefix(l)
                    with self.lock:
                        self.raw_logs.append(clean)
                        added, last_line = True, clean
                    if any(p in clean.lower() for p in _OOM_PATTERNS):
                        oom_found = True
            if added:
                with gs.lock:
                    gs.last_log_msg, gs.last_log_ts = last_line, time.time()
                    if oom_found:
                        gs.oom_seen = True
            return added
        except Exception:
            return False

    def _update_gs(self):
        with self.lock:
            recent = list(self.raw_logs)[-200:]
        slots, prog, ready = {}, "Idle", False
        for l in reversed(recent):
            if any(x in l for x in ("HTTP server is listening", "srv  log_server_r")):
                ready = True
                break
            if "warming up the model" in l:
                prog = "Warming Up..."
                break
            if "load_tensors" in l:
                prog = "Loading Tensors..."
                break
        for l in recent:
            if "slot update_slots" in l and "progress =" in l:
                ms, mt, mp = (
                    re.search(r"id\s+(\d+)", l),
                    re.search(r"task\s+(\d+)", l),
                    re.search(r"progress\s*=\s*([0-9.]+)", l),
                )
                if ms and mt and mp:
                    slots[ms.group(1)] = {
                        "task": mt.group(1),
                        "prog": float(mp.group(1)),
                    }
                    if not ready:
                        prog = f"{float(mp.group(1)) * 100:.1f}%"
        with gs.lock:
            gs.slots, gs.prog, gs.is_ready = slots, prog, ready


lm = LogManager()


def status_worker():
    while True:
        gs.refresh()
        time.sleep(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service_env() -> dict:
    return {**os.environ, "TZ": "Europe/Berlin"}


def _run(cmd, **kwargs):
    return subprocess.run(cmd, env=_service_env(), **kwargs)


def _aggressive_cleanup() -> None:
    """Kill llama-server, sync, drop caches, sleep — cleanup before retry."""
    cmd = (
        "killall -9 llama-server || true && "
        "sudo sync && "
        "echo 3 | sudo tee /proc/sys/vm/drop_caches && "
        "sleep 5"
    )
    subprocess.run(
        cmd, shell=True, check=True, env=_service_env(), capture_output=True, text=True
    )


def _aggressive_service_action(action: str) -> None:
    """Force-kill llama-server, drop filesystem caches, daemon-reload, then perform systemctl action."""
    if action not in ("start", "restart"):
        raise ValueError(f"Invalid action: {action!r}. Must be 'start' or 'restart'.")

    _aggressive_cleanup()

    cmd = f"sudo systemctl daemon-reload && sudo systemctl {action} {SERVICE_NAME}"
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            env=_service_env(),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Aggressive service {action} failed (rc={e.returncode}): {e.stderr}"
        )
        raise


def strip_journal_prefix(line: str) -> str:
    # Handle short-iso format: "2026-05-09T02:15:24+0000 hostname process[pid]: message"
    iso_match = re.match(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+\S+\s+\S+\[\d+\]:\s*(.*)",
        line,
    )
    if iso_match:
        ts, msg = iso_match.group(1), iso_match.group(2)
        # Extract only HH:MM from "2026-05-09T02:15:24+0000"
        time_only = ts[11:16]
        return f"{time_only} | {msg}"
    # Fallback for other formats
    m = _JOURNAL_PREFIX.match(line)
    if not m:
        return line.strip()
    prefix, ts_m = m.group(0), re.match(r"^(\w{3})\s+(\d+)\s+([\d:]+)", line)
    if ts_m:
        day, hms = ts_m.group(2).zfill(2), ts_m.group(3).split(":")
        if len(hms) >= 2:
            return f"{day} {hms[0]}:{hms[1]} | {line[len(prefix) :].strip()}"
    return line[len(prefix) :].strip()


def _fmt_bytes(b: int) -> str:
    val: float = b
    for unit in ("B", "Ki", "Mi", "Gi", "Ti"):
        if abs(val) < 1024:
            return f"{val:.0f}{unit}"
        val /= 1024
    return f"{val:.1f}Pi"


def _read_all_meta(text: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(
            rf"^\s*{re.escape(_LLAMA_MANAGER_META_PREFIX)}\s*(\w+)\s*=\s*(.+?)\s*$",
            line,
        )
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return meta


def _extract_execstart_ngl(text: str) -> int:
    for line in text.splitlines():
        if line.lstrip().startswith("ExecStart="):
            m = re.search(r"(^|\s)-ngl\s+(\d+)\b", line)
            if m:
                return int(m.group(2))
    m = re.search(r"(^|\s)-ngl\s+(\d+)\b", text, flags=re.M)
    if m:
        return int(m.group(2))
    return 0


def _extract_execstart_ctx(text: str) -> int:
    for line in text.splitlines():
        if line.lstrip().startswith("ExecStart="):
            m = re.search(r"(^|\s)-c\s+(\d+)\b", line)
            if m:
                return int(m.group(2))
    m = re.search(r"(^|\s)-c\s+(\d+)\b", text, flags=re.M)
    if m:
        return int(m.group(2))
    return 0


def _write_all_meta(updates: Dict[str, Any]) -> None:
    text = Path(SERVICE_FILE).read_text()
    lines = text.splitlines()
    new_lines: List[str] = []
    updated_keys = set()
    for line in lines:
        m = re.match(rf"^\s*{re.escape(_LLAMA_MANAGER_META_PREFIX)}\s*(\w+)\s*=", line)
        if m and m.group(1) in updates:
            new_lines.append(
                f"{_LLAMA_MANAGER_META_PREFIX} {m.group(1)}={updates[m.group(1)]}"
            )
            updated_keys.add(m.group(1))
        else:
            new_lines.append(line)
    if len(updated_keys) < len(updates):
        insert_pos = 0
        for i, line in enumerate(new_lines):
            if line.lstrip().startswith("ExecStart="):
                insert_pos = i
                break
        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.insert(
                    insert_pos, f"{_LLAMA_MANAGER_META_PREFIX} {key}={value}"
                )
                insert_pos += 1
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    if "ngl_start" in updates:
        val = int(updates["ngl_start"])
        new_text, _ = re.subn(
            r"(^ExecStart=.*?-\s*ngl\s+)\d+\b",
            rf"\g<1>{val}",
            new_text,
            count=1,
            flags=re.M,
        )
    if "ctx_start" in updates:
        val = int(updates["ctx_start"])
        new_text, _ = re.subn(
            r"(^ExecStart=.*?-\s*c\s+)\d+\b",
            rf"\g<1>{val}",
            new_text,
            count=1,
            flags=re.M,
        )
    Path(SERVICE_FILE).write_text(new_text)


def _normalize_execstart(text: str) -> str:
    """Collapse backslash-continued ExecStart lines into one logical line."""
    lines = text.splitlines(keepends=True)
    result: List[str] = []
    in_execstart = False
    for line in lines:
        if line.lstrip().startswith("ExecStart="):
            in_execstart = True
            result.append(line.rstrip("\n"))
        elif (
            in_execstart and line.startswith((" ", "\t")) and result[-1].endswith("\\")
        ):
            result[-1] = result[-1][:-1] + " " + line.strip()
        else:
            in_execstart = False
            result.append(line.rstrip("\n"))
    return "\n".join(result)


def _extract_execstart_flag(
    text: str, flag: str, expect_value: bool = True
) -> Optional[str]:
    """Extract a flag's value from ExecStart line.
    Handles: -flag VALUE, --flag=VALUE, --flag 1/0, --flag
    Returns the value as string, or None if not found.
    When expect_value=True (default), only tries value-greedy pattern.
    When expect_value=False, also tries standalone boolean pattern.
    """
    for line in text.splitlines():
        line_stripped = line.lstrip()
        if not line_stripped.startswith("ExecStart="):
            continue
        # Match: -flag VALUE or --flag VALUE or --flag=VALUE
        m = re.search(r"(?:^|\s)" + re.escape(flag) + r"[=\s]+(\S+)", line)
        if m:
            return m.group(1)
        # Match standalone boolean flag: --flag (no value after)
        if not expect_value:
            m = re.search(r"(?:^|\s)" + re.escape(flag) + r"(?:\s|$)", line)
            if m:
                return "true"
    return None


# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------


def detect_cpu_count():
    """Return the number of CPU cores available."""
    return os.cpu_count() or 1


def detect_vram_info():
    """Detect VRAM info from AMD/Intel GPUs.
    Returns dict with 'total_gib', 'used_gib', and 'free_gib'.
    """
    total = 0
    used = 0
    try:
        for d in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            with open(d) as f:
                total += int(f.read().strip())
        for d in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
            with open(d) as f:
                used += int(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        total = 0
        used = 0
    free = total - used
    return {
        "total_gib": round(total / (1024**3), 2),
        "used_gib": round(used / (1024**3), 2),
        "free_gib": round(free / (1024**3), 2),
    }


def detect_ram_info():
    """Detect system RAM from /proc/meminfo.
    Returns dict with 'total_gib', 'available_gib', and 'used_gib'.
    """
    total = 0
    available = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, OSError):
        total = 0
        available = 0
    used = total - available
    return {
        "total_gib": round(total / (1024**3), 2),
        "available_gib": round(available / (1024**3), 2),
        "used_gib": round(used / (1024**3), 2),
    }


def detect_gpu_type():
    """Detect GPU vendor type. Returns one of: 'amd', 'nvidia', 'intel', 'none'."""
    try:
        for d in glob.glob("/sys/class/drm/card*/device/vendor"):
            with open(d) as f:
                vid = f.read().strip()
            if vid == "0x1002":
                return "amd"
            elif vid == "0x10de":
                return "nvidia"
            elif vid == "0x8086":
                return "intel"
    except (FileNotFoundError, OSError):
        pass
    return "none"


def detect_hardware():
    """Detect all hardware information and return as a dict."""
    vram = detect_vram_info()
    ram = detect_ram_info()
    return {
        "cpu_count": detect_cpu_count(),
        "vram_total_gib": vram["total_gib"],
        "vram_free_gib": vram["free_gib"],
        "ram_total_gib": ram["total_gib"],
        "ram_available_gib": ram["available_gib"],
        "gpu_type": detect_gpu_type(),
    }


def recommend_flags(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Recommend optimal llama-server flag values based on detected hardware.

    Args:
        hardware: dict from detect_hardware()

    Returns:
        dict mapping setting key -> recommended value
    """
    cpu: int = hardware.get("cpu_count", 1)
    vram_free: float = hardware.get("vram_free_gib", 0)
    ram_total: float = hardware.get("ram_total_gib", 0)
    ram_available: float = hardware.get("ram_available_gib", 0)
    gpu_type: str = hardware.get("gpu_type", "none")

    ngl = 0
    if vram_free >= 6:
        ngl = 99
    elif vram_free >= 3:
        ngl = 32
    elif vram_free >= 1:
        ngl = 16

    ctx = 2048
    if gpu_type == "none":
        # No GPU: more RAM available for CPU context
        if ram_total >= 16:
            ctx = 16384
        elif ram_total >= 8:
            ctx = 8192
    elif ram_total >= 32:
        ctx = 8192
    elif ram_total >= 16:
        ctx = 4096
    elif ram_total >= 8:
        ctx = 2048

    threads = cpu
    batch = 2048
    ubatch = 512
    mlock = ram_available >= 32
    parallel = 0
    if vram_free >= 4:
        parallel = 4
    elif vram_free >= 2:
        parallel = 2

    rec: dict[str, Any] = {}
    for s in SETTINGS:
        s_default = s.default
        if s.key == "ngl_start":
            rec[s.key] = ngl
        elif s.key == "threads":
            rec[s.key] = threads
        elif s.key == "ctx_start":
            rec[s.key] = ctx
        elif s.key == "mlock":
            rec[s.key] = mlock
        elif s.key == "parallel_slots":
            rec[s.key] = parallel
        elif s.key == "batch_size":
            rec[s.key] = batch
        elif s.key == "ubatch_size":
            rec[s.key] = ubatch
        else:
            rec[s.key] = s_default
    return rec


def read_all_settings(text: str) -> Dict[str, Any]:
    """Read all auto-restart settings from service text.
    Uses meta comments first, then ExecStart flags, then defaults.
    """
    meta = _read_all_meta(text)
    result: Dict[str, Any] = {}
    for s in SETTINGS:
        # Try meta first
        if s.key in meta:
            val = meta[s.key]
        # Try ExecStart flag if the setting maps to one
        elif s.flag:
            exec_val = _extract_execstart_flag(
                text, s.flag, expect_value=(s.type != "bool")
            )
            if exec_val is not None:
                val = exec_val
            else:
                val = s.default
        else:
            val = s.default

        # Cast to correct type
        if s.type == "int":
            try:
                result[s.key] = int(val)
            except (ValueError, TypeError):
                result[s.key] = s.default
        elif s.type == "bool":
            if isinstance(val, bool):
                result[s.key] = val
            else:
                result[s.key] = _parse_bool(str(val))
        else:
            result[s.key] = str(val) if val is not None else s.default

    return result


def write_setting(text: str, key: str, value: str) -> str:
    """Write a single setting to the service text.
    Updates the # llama-manager: meta comment and optionally the ExecStart flag.
    """
    # Find the Setting definition
    setting = None
    for s in SETTINGS:
        if s.key == key:
            setting = s
            break
    if setting is None:
        raise ValueError(f"Unknown setting: {key}")

    lines = text.splitlines()
    new_lines: List[str] = []
    meta_updated = False
    execstart_pos = None

    for i, line in enumerate(lines):
        # Update existing meta comment
        if line.lstrip().startswith(_LLAMA_MANAGER_META_PREFIX):
            m = re.match(
                r"^" + re.escape(_LLAMA_MANAGER_META_PREFIX) + r"\s*(\w+)\s*=", line
            )
            if m and m.group(1) == key:
                new_lines.append(f"{_LLAMA_MANAGER_META_PREFIX} {key}={value}")
                meta_updated = True
                continue

        # Track ExecStart position
        if line.lstrip().startswith("ExecStart="):
            execstart_pos = i

        new_lines.append(line)

    # If meta comment didn't exist, insert it before ExecStart
    if not meta_updated and execstart_pos is not None:
        new_lines.insert(execstart_pos, f"{_LLAMA_MANAGER_META_PREFIX} {key}={value}")
        execstart_pos += 1

    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"

    # Update ExecStart flag if this setting maps to one
    if setting.flag:
        if value == "":
            # Remove flag entirely when value is empty (user chose to disable)
            new_text = re.sub(
                r"\s+" + re.escape(setting.flag) + r"(?:[=\s]+\S+)?",
                "",
                new_text,
                count=0,
            )
            # Also remove meta comment for this key
            meta_pattern = (
                r"^"
                + re.escape(_LLAMA_MANAGER_META_PREFIX)
                + r"\s*"
                + re.escape(key)
                + r"\s*=.*$\n?"
            )
            new_text = re.sub(meta_pattern, "", new_text, flags=re.MULTILINE)
            return new_text
        if setting.type == "bool":
            is_true = _parse_bool(value)
            pattern = r"(?:^|\s)" + re.escape(setting.flag) + r"(?:[=\s]+\S+)?"
            has_flag = bool(re.search(pattern, new_text, re.M))
            if is_true and not has_flag:
                new_text = re.sub(
                    r"(ExecStart=.*?)(\n|\\n|\s*\\)",
                    lambda m: m.group(1) + " " + setting.flag + " 1" + m.group(2),
                    new_text,
                    count=1,
                )
            elif not is_true and has_flag:
                new_text = re.sub(
                    r"\s+" + re.escape(setting.flag) + r"(?:[=\s]+\S+)?",
                    "",
                    new_text,
                    count=1,
                )
        else:
            flag_value = str(value)
            old_pattern = r"(" + re.escape(setting.flag) + r"[=\s]+)\S+"
            old_text = new_text
            new_text = re.sub(old_pattern, r"\g<1>" + flag_value, new_text, count=1)
            if new_text == old_text:
                # Flag not found in ExecStart — normalize multi-line then append
                new_text = _normalize_execstart(new_text)
                execstart_pattern = r"^(ExecStart=.*?)(\s*)$"
                m = re.search(execstart_pattern, new_text, re.M)
                if m:
                    new_text = (
                        new_text[: m.end(1)]
                        + " "
                        + setting.flag
                        + " "
                        + flag_value
                        + new_text[m.end(2) :]
                    )

    return new_text


def get_ram() -> str:
    try:
        out = subprocess.run(["free", "-b"], capture_output=True, text=True).stdout
        parts = out.splitlines()[1].split()
        used = int(parts[2])
        total = int(parts[1])
        return f"{_fmt_bytes(used)}/{_fmt_bytes(total)}"
    except Exception:
        return "N/A"


def get_vram() -> str:
    try:
        files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
        if not files:
            return "N/A"
        total_used = 0.0
        total_tot = 0.0
        has_tot = False
        for file in files:
            with open(file) as f:
                total_used += int(f.read().strip()) / 1024**3
            tot_path = file.replace("used", "total")
            if os.path.exists(tot_path):
                with open(tot_path) as f:
                    total_tot += int(f.read().strip()) / 1024**3
                has_tot = True
        if has_tot:
            return f"{total_used:.2f}/{total_tot:.2f}GiB"
        return f"{total_used:.2f}GiB"
    except Exception:
        return "N/A"


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_ngl_retry_settings(text: str) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "ngl_start": _extract_execstart_ngl(text),
        "ngl_step": 1,
        "retry_on_oom": False,
        "hang_timeout_mins": 15,
        "oom_detection_timeout_secs": 30,
    }
    for line in text.splitlines():
        m = re.match(
            rf"^\s*{re.escape(_LLAMA_MANAGER_META_PREFIX)}\s*(\w+)\s*=\s*(.+?)\s*$",
            line,
        )
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key == "ngl_start":
            settings["ngl_start"] = int(value)
        elif key == "ngl_step":
            settings["ngl_step"] = int(value)
        elif key == "retry_on_oom":
            settings["retry_on_oom"] = _parse_bool(value)
        elif key == "hang_timeout_mins":
            settings["hang_timeout_mins"] = int(value)
        elif key == "oom_detection_timeout_secs":
            settings["oom_detection_timeout_secs"] = int(value)
    return settings


def should_retry_after_oom(retry_on_oom: bool, current_ngl: int, step: int) -> bool:
    return retry_on_oom and current_ngl > 0 and step > 0


def next_retry_ngl(current_ngl: int, step: int) -> int:
    return max(current_ngl - step, 0)


def is_oom_like_failure(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _OOM_PATTERNS)


def rewrite_ngl_retry_settings(text: str, settings: Dict[str, Any]) -> str:
    existing_meta = _read_all_meta(text)
    merged_meta = {
        **existing_meta,
        "ngl_start": str(settings["ngl_start"]),
        "ngl_step": str(settings["ngl_step"]),
        "retry_on_oom": "true" if settings.get("retry_on_oom", True) else "false",
    }
    if "hang_timeout_mins" in settings:
        merged_meta["hang_timeout_mins"] = str(settings["hang_timeout_mins"])
    if "oom_detection_timeout_secs" in settings:
        merged_meta["oom_detection_timeout_secs"] = str(
            settings["oom_detection_timeout_secs"]
        )

    lines = text.splitlines()
    new_lines: List[str] = []
    inserted = False
    for line in lines:
        if line.lstrip().startswith(_LLAMA_MANAGER_META_PREFIX):
            continue
        if not inserted and line.startswith("ExecStart="):
            for key, value in merged_meta.items():
                new_lines.append(f"{_LLAMA_MANAGER_META_PREFIX} {key}={value}")
            inserted = True
        new_lines.append(line)
    if not inserted:
        for key, value in merged_meta.items():
            new_lines.append(f"{_LLAMA_MANAGER_META_PREFIX} {key}={value}")
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    new_text, count = re.subn(
        r"(-ngl\s+)\d+\b", rf"\g<1>{int(settings['ngl_start'])}", new_text, count=1
    )
    if count == 0:
        raise ValueError("Could not find -ngl in service file")
    return new_text


def save_ngl_retry_settings(settings: Dict[str, Any]) -> None:
    text = Path(SERVICE_FILE).read_text()
    Path(SERVICE_FILE).write_text(rewrite_ngl_retry_settings(text, settings))


def daemon_reload_if_needed() -> None:
    try:
        out = subprocess.run(
            ["systemctl", "show", SERVICE_NAME, "--property=NeedDaemonReload"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out == "NeedDaemonReload=yes":
            subprocess.run(["systemctl", "daemon-reload"], check=False)
    except Exception:
        pass


def remove_stale_flags(text: str) -> str:
    """Remove flags from ExecStart that are not tracked in SETTINGS.
    Also removes corresponding llama-manager meta comments for untracked keys.
    Preserves trailing newlines of the input text.
    """
    known_flags: set = {s.flag for s in SETTINGS if s.flag}
    known_value_flags: set = {s.flag for s in SETTINGS if s.flag and s.type != "bool"}
    known_keys: set = {s.key for s in SETTINGS}

    has_trailing_newline = text.endswith("\n")

    # Normalize multi-line ExecStart into single logical line
    text = _normalize_execstart(text)

    # Remove stale meta comments first
    lines = text.splitlines()
    clean_lines: List[str] = []
    for line in lines:
        m = re.match(rf"^\s*{re.escape(_LLAMA_MANAGER_META_PREFIX)}\s*(\w+)\s*=", line)
        if m and m.group(1) not in known_keys:
            continue  # Skip stale meta comment
        clean_lines.append(line)

    # Now handle ExecStart - remove stale flags
    result_lines: List[str] = []
    for line in clean_lines:
        if line.lstrip().startswith("ExecStart="):
            eq_pos = line.index("=")
            prefix = line[: eq_pos + 1]  # "ExecStart="
            rest = line[eq_pos + 1 :]
            tokens = rest.split()
            new_tokens: List[str] = []
            skip_next = False
            previous_was_value_flag = False
            for i, token in enumerate(tokens):
                if skip_next:
                    skip_next = False
                    continue
                if token.startswith("-"):
                    base_flag = token.split("=")[0] if "=" in token else token
                    if base_flag in known_flags:
                        new_tokens.append(token)
                        previous_was_value_flag = base_flag in known_value_flags
                    elif previous_was_value_flag:
                        # This is a value for the previous flag (e.g., -1 for --reasoning-budget)
                        new_tokens.append(token)
                        previous_was_value_flag = False
                    else:
                        # Unknown flag — skip it and its value (if separate token)
                        previous_was_value_flag = False
                        if (
                            "=" not in token
                            and i + 1 < len(tokens)
                            and not tokens[i + 1].startswith("-")
                        ):
                            skip_next = True
                else:
                    new_tokens.append(token)
                    previous_was_value_flag = False
            result_lines.append(prefix + " ".join(new_tokens))
        else:
            result_lines.append(line)

    result = "\n".join(result_lines)
    if has_trailing_newline:
        result += "\n"
    return result


# ---------------------------------------------------------------------------
# UI Components & View Modes
# ---------------------------------------------------------------------------

(
    COL_BAR,
    COL_SEL,
    COL_NORMAL,
    COL_GREEN,
    COL_RED,
    COL_PURPLE,
    COL_YELLOW,
    COL_CYAN,
    COL_SELBG,
    COL_GREY,
) = range(1, 11)


def draw_status_bar(stdscr, cols):
    with gs.lock:
        s, r, v, p, n, ready, err = (
            gs.service_status,
            gs.ram,
            gs.vram,
            gs.prog,
            gs.ngl,
            gs.is_ready,
            gs.critical_error,
        )
        ctx, o, l, h = (
            gs.ctx,
            gs.meta.get("oom_restart_count", "0"),
            gs.meta.get("loop_restart_count", "0"),
            gs.meta.get("hang_recovery_count", "0"),
        )

    if err:
        stdscr.attron(curses.color_pair(COL_RED) | curses.A_BOLD)
        try:
            stdscr.addstr(0, 0, err.center(cols))
            stdscr.addstr(1, 0, "Press 'r' to reset error state".center(cols))
        except curses.error:
            pass
        stdscr.attroff(curses.color_pair(COL_RED) | curses.A_BOLD)
        return

    def ln(y, lt, rt):
        al = max(0, cols - len(rt))
        stdscr.attron(curses.color_pair(COL_BAR) | curses.A_BOLD)
        try:
            stdscr.addstr(y, 0, (lt[:al].ljust(al) + rt)[:cols])
        except curses.error:
            pass
        stdscr.attroff(curses.color_pair(COL_BAR) | curses.A_BOLD)

    ln(
        0,
        f" llama.service [{s}] | ngl: {n} | ctx: {ctx} | OOM: {o} | Loop: {l} | Hang: {h}",
        f" RAM: {r} ",
    )
    ln(
        1,
        f" Status: {p}" + (" [LOADING]" if not ready and s == "RUNNING" else ""),
        f" VRAM: {v} ",
    )


def run_logs(stdscr):
    stdscr.clear()
    stdscr.timeout(100)
    stdscr.idlok(True)
    stdscr.idcok(True)
    sp, hs, lc, lw, p, dl = -1, 0, -1, None, None, []
    while True:
        rows, cols = stdscr.getmaxyx()
        vh = rows - 4
        with gs.lock:
            wr = gs.wrap_mode
        if cols != lc or wr != lw or lm.new_data_event.is_set():
            if cols != lc:
                stdscr.clear()
            lm.new_data_event.clear()
            lc, lw, dl = cols, wr, []
            with lm.lock:
                raw = list(lm.raw_logs)
            max_w = cols
            for r in raw:
                if wr:
                    for s in range(0, len(r), cols - 1):
                        dl.append(r[s : s + cols - 1])
                else:
                    dl.append(r)
                    max_w = max(max_w, len(r) + 5)
            p = curses.newpad(max(len(dl), vh + 1), max_w)
            for i, l in enumerate(dl):
                try:
                    p.addstr(i, 0, l)
                except curses.error:
                    pass
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            break
        elif k == ord("r"):
            gs.critical_error = None
        elif k == curses.KEY_UP:
            sp = max(0, (sp if sp != -1 else len(dl) - vh) - 1)
        elif k == curses.KEY_DOWN:
            sp = sp + 1 if sp != -1 else -1
            sp = -1 if sp >= len(dl) - vh else sp
        elif k == curses.KEY_LEFT:
            hs = max(0, hs - 5)
        elif k == curses.KEY_RIGHT:
            hs += 5
        elif k in (ord("w"), ord("W")):
            with gs.lock:
                gs.wrap_mode = not gs.wrap_mode
                wr = gs.wrap_mode
            lm.new_data_event.set()
        draw_status_bar(stdscr, cols)
        as_ = max(0, len(dl) - vh) if sp == -1 else min(sp, max(0, len(dl) - vh))
        stdscr.noutrefresh()
        if p and hasattr(p, "pnoutrefresh"):
            try:
                p.pnoutrefresh(as_, hs if not wr else 0, 3, 0, 3 + vh - 1, cols - 1)  # type: ignore[attr-defined]
            except curses.error:
                pass
        curses.doupdate()


def show_message(stdscr, title: str, lines: List[str], wait_key: bool = True) -> None:
    stdscr.nodelay(False)
    stdscr.timeout(-1)
    stdscr.erase()
    rows, cols = stdscr.getmaxyx()
    draw_status_bar(stdscr, cols)
    try:
        stdscr.attron(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
        stdscr.addstr(2, 0, f" === {title} === "[:cols])
        stdscr.attroff(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
    except curses.error:
        pass
    for i, line in enumerate(lines):
        if 4 + i >= rows - 1:
            break
        try:
            stdscr.addstr(4 + i, 0, line[: cols - 1])
        except curses.error:
            pass
    if wait_key:
        try:
            stdscr.attron(curses.color_pair(COL_YELLOW))
            stdscr.addstr(rows - 1, 0, " Any key = continue "[:cols])
            stdscr.attroff(curses.color_pair(COL_YELLOW))
        except curses.error:
            pass
    stdscr.refresh()
    stdscr.getch()
    stdscr.nodelay(True)
    stdscr.timeout(1000)


def _prompt_input(stdscr, title: str, prompt: str, initial: str = "") -> Optional[str]:
    curses.echo()
    curses.curs_set(1)
    stdscr.nodelay(False)
    stdscr.timeout(-1)
    value = initial
    try:
        while True:
            stdscr.erase()
            rows, cols = stdscr.getmaxyx()
            draw_status_bar(stdscr, cols)
            stdscr.attron(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
            stdscr.addstr(2, 0, f" === {title} === "[:cols])
            stdscr.attroff(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
            stdscr.addstr(4, 0, prompt[:cols])
            stdscr.addstr(6, 0, "> ")
            stdscr.addstr(6, 2, value[: max(0, cols - 3)])
            stdscr.addstr(rows - 1, 0, " Enter = confirm    Esc = cancel "[:cols])
            stdscr.move(6, min(cols - 1, 2 + len(value)))
            stdscr.refresh()
            key = stdscr.getch()
            if key == 27:
                return None
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return value.strip()
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                value = value[:-1]
            elif 32 <= key <= 126:
                value += chr(key)
    finally:
        curses.noecho()
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(1000)


def _get_gguf_files() -> tuple[List[str], List[str]]:
    """Return sorted list of .gguf basenames from GGUF_DIR with full paths."""
    if not GGUF_DIR.is_dir():
        return [], []
    paths = sorted(GGUF_DIR.glob("*.gguf"))
    return [p.name for p in paths], [str(p) for p in paths]


def _toggle_setting(stdscr, setting: Setting, current_value: Any) -> str:
    """SPACE toggle: enable/disable, cycle enum, or cycle gguf files. No input prompt."""
    cv = str(current_value)
    if setting.type == "bool":
        new_val = "false" if cv.lower() in ("true", "1", "yes") else "true"
    elif setting.type == "enum":
        options = setting.options
        if not options:
            return cv
        current_idx = options.index(cv) if cv in options else 0
        new_idx = (current_idx + 1) % len(options)
        new_val = options[new_idx]
    elif setting.key == "model_path":
        # Cycle through .gguf files in ~/.gguf
        names, paths = _get_gguf_files()
        if not names:
            # No gguf files found — simple toggle
            new_val = str(setting.default) if cv == "" else ""
        else:
            current_basename = Path(cv).name if cv else ""
            if current_basename in names:
                idx = names.index(current_basename)
                if idx + 1 < len(names):
                    new_val = paths[idx + 1]
                else:
                    new_val = ""  # past last → disable
            else:
                new_val = paths[0]  # first file
    else:
        # int/string: toggle between disabled (empty) and enabled (default)
        if cv == "":
            new_val = str(setting.default)
        else:
            new_val = ""
    # Visual feedback
    if setting.key == "model_path" and new_val:
        display = Path(new_val).name
    else:
        display = _format_display_val(setting, new_val)
    stdscr.erase()
    stdscr.addstr(0, 0, f"  {setting.label}: {display}  ")
    stdscr.clrtoeol()
    stdscr.refresh()
    curses.napms(500)
    return new_val


def _format_display_val(setting: Setting, value: str) -> str:
    """Format a setting value for display (enabled/disabled for bools, etc)."""
    if setting.type == "bool":
        return "enabled" if str(value).lower() in ("true", "1", "yes") else "disabled"
    if value == "":
        return "disabled"
    return value


def _prompt_setting(stdscr, setting: Setting, current_value: Any) -> Optional[str]:
    """ENTER: edit a setting (toggle for bool/enum, input prompt for int/string)."""
    cv = str(current_value)
    if setting.type == "bool":
        new_val = "false" if cv.lower() in ("true", "1", "yes") else "true"
        display = _format_display_val(setting, new_val)
        stdscr.erase()
        stdscr.addstr(0, 0, f"  {setting.label}: {display}  ")
        stdscr.clrtoeol()
        stdscr.refresh()
        curses.napms(500)
        return new_val
    elif setting.type == "int":
        return _prompt_input(
            stdscr,
            "Restart & Timeout Settings",
            f"Enter new {setting.key} (empty=disable):",
            cv,
        )
    elif setting.type == "string":
        return _prompt_input(
            stdscr,
            setting.label,
            setting.label + " (empty=disable):",
            cv,
        )
    elif setting.type == "enum":
        options = setting.options
        if not options:
            return cv
        current_idx = options.index(cv) if cv in options else 0
        new_idx = (current_idx + 1) % len(options)
        new_value = options[new_idx]
        # Visual feedback for cycling
        stdscr.erase()
        stdscr.addstr(0, 0, f"  {setting.label}: {new_value}  ")
        stdscr.clrtoeol()
        stdscr.refresh()
        curses.napms(500)
        return new_value
    return cv


def _apply_setting_change(
    stdscr, display_entries, selected, current_settings, *, use_toggle=False
):
    """Apply a setting change: toggle (SPACE) or edit prompt (ENTER)."""
    entry = display_entries[selected]
    if entry is None:
        return True  # signal to break
    if entry == "__SEPARATOR__":
        return False
    if entry == "__AUTO_DETECT__":
        try:
            hardware = detect_hardware()
            recommendations = recommend_flags(hardware)
            text = Path(SERVICE_FILE).read_text()
            for key, val in recommendations.items():
                str_val = str(val).lower() if isinstance(val, bool) else str(val)
                text = write_setting(text, key, str_val)
            Path(SERVICE_FILE).write_text(text)
            current_settings.clear()
            current_settings.update(read_all_settings(text))
            show_message(
                stdscr,
                "Auto-Detect & Recommend",
                ["Settings applied!"]
                + [f"  {k} = {v}" for k, v in recommendations.items()],
            )
        except Exception as e:
            show_message(stdscr, "Error", [f"Auto-detect failed: {e}"])
        return False
    current_val = current_settings.get(entry.key, entry.default)
    try:
        if use_toggle:
            new_val = _toggle_setting(stdscr, entry, current_val)
        else:
            new_val = _prompt_setting(stdscr, entry, current_val)
        if new_val is None:
            return False
        text = Path(SERVICE_FILE).read_text()
        new_text = write_setting(text, entry.key, new_val)
        Path(SERVICE_FILE).write_text(new_text)
        if new_val == "":
            current_settings[entry.key] = ""
            logging.info(f"Setting {entry.key} -> disabled")
        elif entry.type == "bool":
            current_settings[entry.key] = _parse_bool(new_val)
            logging.info(f"Setting {entry.key} -> {new_val}")
        else:
            current_settings[entry.key] = (
                int(new_val) if entry.type == "int" else new_val
            )
            logging.info(f"Setting {entry.key} -> {new_val}")
    except Exception as e:
        show_message(stdscr, "Settings Error", [f"Failed to save {entry.key}: {e}"])
        logging.error(f"Failed to save setting {entry.key}: {e}")
    return False


def run_auto_restart_settings(stdscr) -> None:
    """Submenu showing individual auto-restart settings entries."""
    try:
        current_settings = read_all_settings(Path(SERVICE_FILE).read_text())
    except Exception as e:
        show_message(stdscr, "Settings Error", [f"Failed to read service file: {e}"])
        return

    selected = 0
    display_entries: list[Any] = list(SETTINGS)
    display_entries.append("__SEPARATOR__")
    display_entries.append("__AUTO_DETECT__")
    display_entries.append(None)  # sentinel for "Back"

    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()

        title = "=== Restart & Timeout Settings ==="
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(0, max(0, (cols - len(title)) // 2), title)
        stdscr.attroff(curses.color_pair(1))

        stdscr.addstr(2, 2, f"{'Setting':30s} {'Status':>15s}")
        stdscr.addstr(3, 2, "-" * 48)

        visible_count = min(rows - 6, len(display_entries))
        start_idx = max(0, selected - visible_count // 2)
        end_idx = min(start_idx + visible_count, len(display_entries))

        for i in range(start_idx, end_idx):
            y = 4 + i - start_idx
            entry = display_entries[i]

            if entry is None:
                indicator = " <-" if i == selected else "   "
                stdscr.addstr(y, 2, f"[q] Back to main menu{indicator}")
                continue

            if entry == "__SEPARATOR__":
                stdscr.addstr(y, 2, "─" * 48)
                continue

            if entry == "__AUTO_DETECT__":
                cursor = ">" if i == selected else " "
                line = f"{cursor}   ⚡ Auto-Detect & Recommend"
                if i == selected:
                    stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                    stdscr.addstr(y, 2, line[: cols - 2])
                    stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
                else:
                    stdscr.attron(curses.color_pair(COL_GREEN))
                    stdscr.addstr(y, 2, line[: cols - 2])
                    stdscr.attroff(curses.color_pair(COL_GREEN))
                continue

            current_str = str(current_settings.get(entry.key, entry.default))
            if entry.type == "bool":
                is_enabled = str(current_str).lower() in ("true", "1", "yes")
                display_val = "enabled" if is_enabled else "disabled"
            else:
                is_enabled = current_str != ""
                if current_str != "":
                    display_val = (
                        Path(current_str).name
                        if entry.key == "model_path"
                        else current_str
                    )
                else:
                    display_val = "disabled"

            checkbox = "[X]" if is_enabled else "[ ]"
            cursor = ">" if i == selected else " "
            line = f"{cursor} {checkbox} {entry.key:28s} {display_val:>14s}"

            if i == selected:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(y, 2, line[: cols - 2])
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            elif not is_enabled:
                stdscr.attron(curses.color_pair(COL_GREY))
                stdscr.addstr(y, 2, line[: cols - 2])
                stdscr.attroff(curses.color_pair(COL_GREY))
            else:
                stdscr.addstr(y, 2, line[: cols - 2])

        stdscr.addstr(rows - 2, 2, "UP/DOWN nav | ENTER edit | SPACE toggle | q back")
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (curses.KEY_UP, ord("k")):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = min(len(display_entries) - 1, selected + 1)
        elif key in (curses.KEY_ENTER, 10, 13):
            if _apply_setting_change(
                stdscr, display_entries, selected, current_settings, use_toggle=False
            ):
                break
        elif key == ord(" "):
            if _apply_setting_change(
                stdscr, display_entries, selected, current_settings, use_toggle=True
            ):
                break


def run_ngl_retry_settings(stdscr) -> None:
    try:
        current = read_ngl_retry_settings(Path(SERVICE_FILE).read_text())
    except Exception as e:
        show_message(
            stdscr, "Restart & Timeout Settings", [f"Failed to read service file: {e}"]
        )
        return
    ngl_start = _prompt_input(
        stdscr,
        "Restart & Timeout Settings",
        "Starting ngl value:",
        str(current["ngl_start"]),
    )
    if ngl_start is None:
        return
    ngl_step = _prompt_input(
        stdscr,
        "Restart & Timeout Settings",
        "ngl decrement per retry:",
        str(current["ngl_step"]),
    )
    if ngl_step is None:
        return
    retry_default = "Y" if current["retry_on_oom"] else "n"
    retry_on_oom = _prompt_input(
        stdscr,
        "Restart & Timeout Settings",
        "Retry with lower ngl after OOM crash? [Y/n]",
        retry_default,
    )
    if retry_on_oom is None:
        return
    hang_timeout = _prompt_input(
        stdscr,
        "Restart & Timeout Settings",
        "Hang timeout (minutes):",
        str(current.get("hang_timeout_mins", 15)),
    )
    if hang_timeout is None:
        return
    oom_timeout = _prompt_input(
        stdscr,
        "Restart & Timeout Settings",
        "OOM detection timeout (seconds):",
        str(current.get("oom_detection_timeout_secs", 30)),
    )
    if oom_timeout is None:
        return
    try:
        settings = {
            "ngl_start": int(ngl_start),
            "ngl_step": int(ngl_step),
            "retry_on_oom": _parse_bool(retry_on_oom),
            "hang_timeout_mins": int(hang_timeout) if hang_timeout else 15,
            "oom_detection_timeout_secs": int(oom_timeout) if oom_timeout else 30,
        }
        save_ngl_retry_settings(settings)
        daemon_reload_if_needed()
        logging.info(
            f"Recovery settings saved: ngl_start={settings['ngl_start']}, "
            f"step={settings['ngl_step']}, retry={settings['retry_on_oom']}, "
            f"hang_timeout={settings.get('hang_timeout_mins')}m, "
            f"oom_timeout={settings.get('oom_detection_timeout_secs')}s"
        )
    except Exception as e:
        show_message(
            stdscr, "Restart & Timeout Settings", [f"Failed to save settings: {e}"]
        )
        logging.error(f"Failed to save recovery settings: {e}")
        return
    show_message(
        stdscr,
        "Restart & Timeout Settings",
        [
            f"Saved starting ngl: {settings['ngl_start']}",
            f"Saved decrement: {settings['ngl_step']}",
            f"Retry on OOM: {'yes' if settings['retry_on_oom'] else 'no'}",
            f"Hang timeout: {settings.get('hang_timeout_mins', 15)} min",
            f"OOM detection: {settings.get('oom_detection_timeout_secs', 30)} sec",
        ],
    )


def _get_recent_logs(n: int = 10) -> List[str]:
    try:
        out = _run(
            [
                "journalctl",
                "-u",
                SERVICE_NAME,
                "-n",
                "100",
                "-o",
                "short-iso",
                "--show-cursor",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
        ).stdout
        lines = [strip_journal_prefix(l) for l in out.strip().splitlines() if l.strip()]
        return [l for l in lines if l][-n:]
    except Exception:
        return []


def _start_service_with_ngl_retry(stdscr, action: str) -> None:
    try:
        text = Path(SERVICE_FILE).read_text()
        text = remove_stale_flags(text)
        Path(SERVICE_FILE).write_text(text)
        settings = read_ngl_retry_settings(text)
    except Exception as e:
        show_message(
            stdscr, "Restart & Timeout Settings", [f"Failed to read service file: {e}"]
        )
        return
    current_ngl = int(settings["ngl_start"])
    step = int(settings["ngl_step"])
    timeout_secs = int(settings.get("oom_detection_timeout_secs", 30))
    logging.info(
        f"Service {action} starting: ngl={current_ngl}, step={step}, oom_timeout={timeout_secs}s"
    )
    while True:
        settings["ngl_start"] = current_ngl
        try:
            save_ngl_retry_settings(settings)
            _aggressive_service_action(action)
        except Exception as e:
            show_message(
                stdscr,
                "Restart & Timeout Settings",
                [f"Failed to {action} service: {e}"],
            )
            logging.error(f"Service {action} failed: {e}")
            return
        if not settings.get("retry_on_oom", True):
            logging.info(
                f"Service {action} completed (retry disabled), ngl={current_ngl}"
            )
            return
        deadline = time.time() + timeout_secs
        saw_oom = False
        while time.time() < deadline:
            if any(is_oom_like_failure(line) for line in _get_recent_logs(20)):
                saw_oom = True
                break
            if (
                _run(["systemctl", "is-active", "--quiet", SERVICE_NAME]).returncode
                == 0
            ):
                # Service is running successfully
                break
            if (
                _run(["systemctl", "is-failed", "--quiet", SERVICE_NAME]).returncode
                == 0
            ):
                time.sleep(0.5)
                if any(is_oom_like_failure(line) for line in _get_recent_logs(20)):
                    saw_oom = True
                break
            time.sleep(0.5)
        if not saw_oom:
            # Check if service is actually running (not failed for other reasons)
            if (
                _run(["systemctl", "is-active", "--quiet", SERVICE_NAME]).returncode
                != 0
            ):
                show_message(
                    stdscr,
                    "Service Error",
                    [
                        "Service failed to start (non-OOM error).",
                        "Check journal for details.",
                    ],
                )
                logging.error(f"Service {action} failed (non-OOM error)")
                return
            logging.info(f"Service {action} succeeded, ngl={current_ngl}")
            return
        next_ngl = next_retry_ngl(current_ngl, step)
        logging.warning(
            f"OOM detected during {action}, retrying: ngl {current_ngl} -> {next_ngl}"
        )
        if next_ngl == current_ngl:
            show_message(
                stdscr,
                "Restart & Timeout Settings",
                [f"OOM retry stopped at ngl={current_ngl}."],
            )
            logging.error(f"OOM retry exhausted: stuck at ngl={current_ngl}")
            return
        current_ngl = next_ngl
        _aggressive_cleanup()


def draw_menu(stdscr, selected: int, rows: int, cols: int) -> None:
    stdscr.attron(curses.color_pair(COL_NORMAL))
    hint = "  Arrow keys / j k = navigate    Enter = select    q = quit"
    try:
        stdscr.addstr(2, 0, hint[:cols])
    except curses.error:
        pass
    stdscr.attroff(curses.color_pair(COL_NORMAL))

    for i, (label, _) in enumerate(MENU):
        row = 4 + i
        if row >= rows - 1:
            break
        text = f"  {label}  "
        try:
            if i == selected:
                stdscr.attron(curses.color_pair(COL_SEL) | curses.A_BOLD)
                stdscr.addstr(row, 0, text.ljust(30))
                stdscr.attroff(curses.color_pair(COL_SEL) | curses.A_BOLD)
            else:
                stdscr.attron(curses.color_pair(COL_NORMAL))
                stdscr.addstr(row, 0, text.ljust(30))
                stdscr.attroff(curses.color_pair(COL_NORMAL))
        except curses.error:
            pass


def get_free_table():
    headers = ["Type", "Total", "Used", "Free", "Shared", "Buff/Cache", "Avail"]
    try:
        import subprocess

        out = subprocess.run(["free", "-b"], capture_output=True, text=True).stdout
        lines = out.strip().splitlines()
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            label = parts[0].rstrip(":")
            nums = parts[1:]
            if label == "Mem" and len(nums) >= 6:
                rows.append(
                    [
                        label,
                        _fmt_bytes(int(nums[0])),
                        _fmt_bytes(int(nums[1])),
                        _fmt_bytes(int(nums[2])),
                        _fmt_bytes(int(nums[3])),
                        _fmt_bytes(int(nums[4])),
                        _fmt_bytes(int(nums[5])),
                    ]
                )
            elif label == "Swap" and len(nums) >= 2:
                rows.append(
                    [
                        label,
                        _fmt_bytes(int(nums[0])),
                        _fmt_bytes(int(nums[1])),
                        _fmt_bytes(int(nums[2])) if len(nums) > 2 else "0B",
                        "-",
                        "-",
                        "-",
                    ]
                )
        return headers, rows
    except Exception:
        return headers, []


def draw_free_h(stdscr, start_row: int, cols: int) -> int:
    try:
        stdscr.attron(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
        stdscr.addstr(start_row, 0, " === MEMORY === "[:cols])
        stdscr.attroff(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
    except curses.error:
        pass

    headers, rows = get_free_table()
    col_w = [len(h) for h in headers]
    for row in rows:
        for ci, cell in enumerate(row):
            col_w[ci] = max(col_w[ci], len(cell))

    def fmt_row(cells):
        return "  ".join(c.rjust(col_w[i]) for i, c in enumerate(cells))

    header_line = fmt_row(headers)
    data_lines = [fmt_row(r) for r in rows]

    try:
        stdscr.attron(curses.color_pair(COL_CYAN) | curses.A_BOLD)
        stdscr.addstr(start_row + 1, 1, header_line[: cols - 2])
        stdscr.attroff(curses.color_pair(COL_CYAN) | curses.A_BOLD)
    except curses.error:
        pass

    for i, line in enumerate(data_lines):
        try:
            stdscr.attron(curses.color_pair(COL_NORMAL))
            stdscr.addstr(start_row + 2 + i, 1, line[: cols - 2])
            stdscr.attroff(curses.color_pair(COL_NORMAL))
        except curses.error:
            pass
    return start_row + 2 + len(data_lines)


def run_watch(stdscr) -> None:
    try:
        text = Path(SERVICE_FILE).read_text()
        meta = _read_all_meta(text)
        watch_lines = int(meta.get("watch_log_lines", 6))
    except Exception:
        watch_lines = 6
    stdscr.nodelay(True)
    stdscr.timeout(1000)
    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        draw_status_bar(stdscr, cols)
        next_row = draw_free_h(stdscr, 2, cols)
        with lm.lock:
            logs = list(lm.raw_logs)[-watch_lines:]
        try:
            stdscr.attron(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
            stdscr.addstr(next_row, 0, " === RECENT LOGS === "[:cols])
            stdscr.attroff(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
        except curses.error:
            pass
        for i, line in enumerate(logs):
            row = next_row + 1 + i
            if row >= rows - 8:
                break
            try:
                stdscr.attron(curses.color_pair(COL_NORMAL))
                stdscr.addstr(row, 1, strip_journal_prefix(line)[: cols - 2])
                stdscr.attroff(curses.color_pair(COL_NORMAL))
            except curses.error:
                pass
        sep_row = next_row + 1 + len(logs) + 1
        with gs.lock:
            slots = gs.slots
        try:
            stdscr.attron(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
            stdscr.addstr(sep_row, 0, " === SLOT PROGRESS === "[:cols])
            stdscr.attroff(curses.color_pair(COL_PURPLE) | curses.A_BOLD)
        except curses.error:
            pass
        for i, (sid, sdata) in enumerate(slots.items()):
            row = sep_row + 1 + i
            if row >= rows - 1:
                break
            prog = sdata.get("prog", 0.0)
            task = sdata.get("task", "?")
            filled = int(prog * 20)
            bar = "#" * filled + "-" * (20 - filled)
            text = f" Slot {sid} (Task {task}): [{bar}] {prog * 100:.1f}%"
            try:
                stdscr.attron(curses.color_pair(COL_NORMAL))
                stdscr.addstr(row, 1, text[: cols - 2])
                stdscr.attroff(curses.color_pair(COL_NORMAL))
            except curses.error:
                pass


def run_action(stdscr, action: str) -> None:
    if action == "watch":
        run_watch(stdscr)
    elif action == "logs":
        run_logs(stdscr)
    elif action == "reset_error":
        ngl_setting = next((s for s in SETTINGS if s.key == "ngl_start"), None)
        if ngl_setting is None:
            raise KeyError("ngl_start not found in SETTINGS")
        ngl_default = ngl_setting.default
        ctx_setting = next((s for s in SETTINGS if s.key == "ctx_start"), None)
        if ctx_setting is None:
            raise KeyError("ctx_start not found in SETTINGS")
        ctx_default = ctx_setting.default
        with gs.lock:
            gs.last_manual_reset = time.time()
            gs.critical_error = None
            gs.oom_seen = False
            gs.loading_since = None
            gs.last_log_ts = None
            gs.last_log_msg = ""
            gs.is_ready = False
            _write_all_meta(
                {
                    "oom_restart_count": "0",
                    "hang_recovery_count": "0",
                    "loop_restart_count": "0",
                    "last_recovery_reason": "",
                    "ngl_start": str(ngl_default),
                    "ctx_start": str(ctx_default),
                }
            )
        logging.info(
            "User action: reset_error, full recovery state reset (counters, ngl_start, ctx_start)"
        )
        show_message(
            stdscr,
            "Error Reset",
            [
                "All recovery state has been reset:",
                "",
                "  • Critical error cleared",
                "  • OOM detection counter reset",
                "  • Loading state reset",
                "  • Recovery counters reset to 0",
                "  • ngl_start reset to {}".format(ngl_default),
                "  • ctx_start reset to {}".format(ctx_default),
                "",
                "Press any key to return to menu.",
            ],
            wait_key=True,
        )
    elif action == "auto_restart_settings":
        run_auto_restart_settings(stdscr)
    elif action == "start":
        logging.info("User action: start")
        show_message(
            stdscr,
            "Starting Service",
            [
                "Starting llama.service with cache cleanup...",
                "",
                "This will take a few seconds.",
            ],
            wait_key=True,
        )
        daemon_reload_if_needed()
        _start_service_with_ngl_retry(stdscr, "start")
    elif action == "stop":
        logging.info("User action: stop")
        daemon_reload_if_needed()
        _run(["systemctl", "stop", SERVICE_NAME])
    elif action == "restart":
        logging.info("User action: restart")
        show_message(
            stdscr,
            "Restarting Service",
            [
                "Restarting llama.service with cache cleanup...",
                "",
                "This will take a few seconds.",
            ],
            wait_key=True,
        )
        daemon_reload_if_needed()
        _start_service_with_ngl_retry(stdscr, "restart")
    elif action == "journal":
        curses.endwin()
        try:
            subprocess.run(
                [
                    "journalctl",
                    "-u",
                    SERVICE_NAME,
                    "-f",
                    "-o",
                    "short-iso",
                    "--no-pager",
                ]
            )
        except KeyboardInterrupt:
            pass
        finally:
            stdscr.refresh()


def main(stdscr) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(COL_BAR, curses.COLOR_WHITE, 92)
    curses.init_pair(COL_SEL, curses.COLOR_WHITE, 92)
    curses.init_pair(COL_NORMAL, curses.COLOR_WHITE, -1)
    curses.init_pair(COL_GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(COL_RED, curses.COLOR_RED, -1)
    curses.init_pair(COL_PURPLE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COL_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(COL_CYAN, curses.COLOR_CYAN, -1)
    curses.init_pair(COL_SELBG, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(COL_GREY, 8, -1)  # dark grey for disabled entries

    threading.Thread(target=status_worker, daemon=True).start()
    lm.start()

    selected = 0
    stdscr.timeout(1000)

    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        draw_status_bar(stdscr, cols)
        draw_menu(stdscr, selected, rows, cols)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        elif key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(MENU)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(MENU)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            action = MENU[selected][1]
            if action == "quit":
                break
            run_action(stdscr, action)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception("Fatal error")
        print(f"Fatal error: {e}", file=__import__("sys").stderr)
