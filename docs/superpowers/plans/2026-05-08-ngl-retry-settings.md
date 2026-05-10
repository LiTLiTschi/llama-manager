# ngl Retry Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TUI menu to configure a starting `-ngl` value, a decrement step, and an optional retry-on-OOM toggle that automatically restarts `llama.service` with a lower `-ngl` after OOM-like crashes.

**Architecture:** Keep the current systemd unit file as the single source of truth. The TUI will read and rewrite `ExecStart` plus a small block of comment metadata in `llama.service` that stores the retry settings. The retry loop will watch the journal for OOM-style exits and, when enabled, rewrite `-ngl` to `max(current_ngl - step, 0)`, reload systemd, and restart the service until the floor is reached.

**Tech Stack:** Python 3.10+, stdlib `curses`, `subprocess`, `re`, `pathlib`, `unittest`, `systemd`, `journalctl`

---

### Task 1: Add failing tests for ngl settings

**Files:**
- Modify: `tests/test_llama_manager.py`

- [ ] **Step 1: Write the failing test**

Add tests that lock in:

```python
def test_menu_includes_ngl_retry_settings_entry(self):
    self.assertIn(("ngl Retry Settings", "ngl_retry_settings"), llama_manager.MENU)

def test_retry_settings_round_trip_in_service_text(self):
    text = """[Service]
# llama-manager: ngl_start=24
# llama-manager: ngl_step=4
# llama-manager: retry_on_oom=true
ExecStart=/home/liu/llama.cpp/build/bin/llama-server -m /x.gguf -ngl 24
"""
    settings = llama_manager.read_ngl_retry_settings(text)
    self.assertEqual(settings["ngl_start"], 24)
    self.assertEqual(settings["ngl_step"], 4)
    self.assertTrue(settings["retry_on_oom"])

def test_should_retry_oom_requires_enabled_toggle_and_positive_next_ngl(self):
    self.assertTrue(llama_manager.should_retry_after_oom(True, 24, 4))
    self.assertFalse(llama_manager.should_retry_after_oom(False, 24, 4))
    self.assertFalse(llama_manager.should_retry_after_oom(True, 0, 4))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/liu/gists/llama-manager
python3 -m unittest tests.test_llama_manager -v
```

Expected: fail because the new helper functions and menu entry do not exist yet.

- [ ] **Step 3: Do not implement code yet**

Stop after confirming the failures so the next task can implement only the missing helpers.

---

### Task 2: Add TUI entry and settings editor

**Files:**
- Modify: `llama-manager.py`
- Modify: `tests/test_llama_manager.py`

- [ ] **Step 1: Extend the menu test to cover the new action wiring**

Add a test that `run_action(stdscr, "ngl_retry_settings")` reaches a dedicated settings screen function instead of falling through.

- [ ] **Step 2: Implement the smallest UI surface**

Add:

```python
("ngl Retry Settings", "ngl_retry_settings")
```

to `MENU`, then add a `run_ngl_retry_settings(stdscr)` screen that:

1. shows the current `ngl_start`, `ngl_step`, and toggle state
2. lets the user edit the two integers with `prompt_input`
3. lets the user toggle retry-on-OOM with a simple yes/no prompt
4. writes the updated settings back into `SERVICE_FILE`

- [ ] **Step 3: Run the targeted tests**

Run:

```bash
cd /home/liu/gists/llama-manager
python3 -m unittest tests.test_llama_manager.JournalActionTest -v
```

Expected: the new menu item test passes and the action dispatch test passes.

---

### Task 3: Implement OOM retry logic in the service flow

**Files:**
- Modify: `llama-manager.py`
- Modify: `tests/test_llama_manager.py`

- [ ] **Step 1: Add a failing retry test**

Add a test that simulates a service stop after an OOM-like journal line and verifies the next `-ngl` value is decremented and bounded at zero.

```python
def test_oom_retry_decrements_ngl_and_stops_at_zero(self):
    self.assertEqual(llama_manager.next_retry_ngl(24, 4), 20)
    self.assertEqual(llama_manager.next_retry_ngl(3, 4), 0)
```

Also add a log-pattern test for OOM detection based on the existing abort message:

```python
def test_detect_oom_like_failure(self):
    self.assertTrue(llama_manager.is_oom_like_failure(
        "common_fit_params: failed to fit params to free device memory"
    ))
```

- [ ] **Step 2: Implement the retry path**

Add helpers to:

1. parse the current `-ngl` value from `ExecStart`
2. detect OOM-like crashes from the journal
3. compute the next `-ngl`
4. rewrite the unit file and restart only when retry is enabled

Keep the retry behavior bounded:

```python
next_ngl = max(current_ngl - step, 0)
```

and stop once `next_ngl == 0`.

- [ ] **Step 3: Run the retry tests**

Run:

```bash
cd /home/liu/gists/llama-manager
python3 -m unittest tests.test_llama_manager -v
```

Expected: all tests pass.

---

### Task 4: Update user-facing docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short feature note**

Document that the TUI now includes `ngl Retry Settings`, that settings live in `llama.service`, and that OOM retries step `-ngl` down until zero.

- [ ] **Step 2: Sanity-check the wording**

Make sure the README matches the actual UI labels and does not describe unsupported behavior.

---

### Task 5: Final verification

**Files:**
- None

- [ ] **Step 1: Run the full test suite**

Run:

```bash
cd /home/liu/gists/llama-manager
python3 -m unittest discover -s tests -v
```

- [ ] **Step 2: Smoke-test the TUI entry points**

Start the app and verify the menu shows the new settings entry and the settings screen opens without crashing.
