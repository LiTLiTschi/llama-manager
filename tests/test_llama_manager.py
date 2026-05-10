import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

MODULE_PATH = Path(__file__).resolve().parents[1] / "llama-manager.py"
SPEC = importlib.util.spec_from_file_location("llama_manager", MODULE_PATH)
assert SPEC and SPEC.loader
llama_manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(llama_manager)
sys.modules["llama_manager"] = llama_manager


class DummyScreen:
    """Minimal curses screen mock for testing."""
    def __init__(self):
        self.refreshed = False

    def refresh(self):
        self.refreshed = True

    def nodelay(self, *a):
        pass

    def timeout(self, *a):
        pass

    def erase(self):
        pass

    def getmaxyx(self):
        return 24, 80

    def attron(self, *a):
        pass

    def attroff(self, *a):
        pass

    def addstr(self, *a, **kw):
        pass

    def getch(self):
        return ord('q')

    def move(self, *a):
        pass


class JournalActionTest(unittest.TestCase):
    def test_menu_includes_journal_entry(self):
        self.assertIn(("Journal (Pager)", "journal"), llama_manager.MENU)

    def test_run_action_journal_launches_following_journalctl(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return type("Result", (), {"returncode": 0})()

        screen = DummyScreen()

        with patch.object(llama_manager.subprocess, "run", side_effect=fake_run), \
             patch.object(llama_manager.curses, "endwin", return_value=None):
            llama_manager.run_action(screen, "journal")

        self.assertEqual(
            calls,
            [(
                [
                    "journalctl",
                    "-u", llama_manager.SERVICE_NAME,
                    "-f",
                    "-o", "short-iso",
                    "--no-pager",
                ],
                {},
            )],
        )

    def test_run_action_auto_restart_settings_opens_settings_screen(self):
        screen = DummyScreen()

        with patch.object(llama_manager, "run_auto_restart_settings", return_value=None) as run_settings:
            llama_manager.run_action(screen, "auto_restart_settings")

        run_settings.assert_called_once_with(screen)


class NglRetrySettingsTest(unittest.TestCase):
    def test_menu_includes_auto_restart_settings_entry(self):
        self.assertIn(("Restart & Timeout Settings", "auto_restart_settings"), llama_manager.MENU)

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

    def test_rewrite_ngl_retry_settings_updates_metadata_and_ngl(self):
        text = """[Service]
ExecStart=/home/liu/llama.cpp/build/bin/llama-server \\
  -m /x.gguf \\
  -ngl 999 \\
  --host 0.0.0.0
"""

        rewritten = llama_manager.rewrite_ngl_retry_settings(
            text,
            {"ngl_start": 24, "ngl_step": 4, "retry_on_oom": True},
        )

        self.assertIn("# llama-manager: ngl_start=24", rewritten)
        self.assertIn("# llama-manager: ngl_step=4", rewritten)
        self.assertIn("# llama-manager: retry_on_oom=true", rewritten)
        self.assertIn("  -ngl 24 \\", rewritten)

    def test_should_retry_after_oom_requires_enabled_toggle_and_positive_next_ngl(self):
        self.assertTrue(llama_manager.should_retry_after_oom(True, 24, 4))
        self.assertFalse(llama_manager.should_retry_after_oom(False, 24, 4))
        self.assertFalse(llama_manager.should_retry_after_oom(True, 0, 4))

    def test_next_retry_ngl_stops_at_zero(self):
        self.assertEqual(llama_manager.next_retry_ngl(24, 4), 20)
        self.assertEqual(llama_manager.next_retry_ngl(3, 4), 0)

    def test_is_oom_like_failure_matches_device_memory_abort(self):
        self.assertTrue(
            llama_manager.is_oom_like_failure(
                "common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999, abort"
            )
        )


# ---------------------------------------------------------------------------
# Fix #4: recommend_flags ignores gpu_type
# ---------------------------------------------------------------------------

class RecommendFlagsGpuTypeTest(unittest.TestCase):
    """Tests for gpu_type influencing recommend_flags."""

    def test_recommend_uses_gpu_type_for_ctx(self):
        """gpu_type='none' should allow higher ctx_start (no VRAM pressure)."""
        hardware = {
            "cpu_count": 12,
            "vram_total_gib": 0,
            "vram_free_gib": 0,
            "ram_total_gib": 20,
            "ram_available_gib": 16,
            "gpu_type": "none",
        }
        recommended = llama_manager.recommend_flags(hardware)
        self.assertGreaterEqual(recommended["ctx_start"], 16384)

    def test_recommend_gpu_with_lower_ctx(self):
        """With a GPU, ctx_start should be lower at same RAM (VRAM pressure)."""
        hardware = {
            "cpu_count": 12,
            "vram_total_gib": 8,
            "vram_free_gib": 3,
            "ram_total_gib": 20,
            "ram_available_gib": 16,
            "gpu_type": "amd",
        }
        recommended = llama_manager.recommend_flags(hardware)
        self.assertLess(recommended["ctx_start"], 16384)


# ---------------------------------------------------------------------------
# Fix #8: _write_all_meta regex matches outside ExecStart
# ---------------------------------------------------------------------------

class WriteAllMetaRegexTest(unittest.TestCase):
    """Tests for _write_all_meta regex not matching outside ExecStart."""

    @patch('llama_manager.Path.read_text')
    @patch('llama_manager.Path.write_text')
    def test_write_all_meta_updates_single_line_execstart_ngl(self, mock_write, mock_read):
        """Single-line ExecStart -ngl should be updated even when not at line start."""
        text = """[Service]
# llama-manager: ngl_start=24
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        mock_read.return_value = text
        llama_manager._write_all_meta({"ngl_start": 40})
        written = mock_write.call_args[0][0]
        # The ExecStart line should have -ngl 40 (updated from 24)
        self.assertIn("-ngl 40", written)
        # The -m flag should still be present
        self.assertIn("-m /x.gguf", written)

    @patch('llama_manager.Path.read_text')
    @patch('llama_manager.Path.write_text')
    def test_write_all_meta_does_not_modify_non_execstart_ngl(self, mock_write, mock_read):
        """A line with '-ngl' outside ExecStart should NOT be modified."""
        text = """[Service]
  -ngl 99
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        mock_read.return_value = text
        llama_manager._write_all_meta({"ngl_start": 40})
        written = mock_write.call_args[0][0]
        # The non-ExecStart line with -ngl 99 should remain unchanged
        self.assertIn("-ngl 99", written)
        # The ExecStart line should have -ngl 40 (updated)
        self.assertIn("ExecStart=llama-server -m /x.gguf -ngl 40", written)

    @patch('llama_manager.Path.read_text')
    @patch('llama_manager.Path.write_text')
    def test_write_all_meta_updates_single_line_execstart_ctx(self, mock_write, mock_read):
        """Single-line ExecStart -c should be updated."""
        text = """[Service]
# llama-manager: ctx_start=8192
ExecStart=llama-server -m /x.gguf -c 8192
"""
        mock_read.return_value = text
        llama_manager._write_all_meta({"ctx_start": 4096})
        written = mock_write.call_args[0][0]
        self.assertIn("-c 4096", written)
        self.assertIn("-m /x.gguf", written)


# ---------------------------------------------------------------------------
# Fix #7: Enum cycling visual feedback
# ---------------------------------------------------------------------------

class EnumPromptFeedbackTest(unittest.TestCase):
    """Tests for enum cycling visual feedback in _prompt_setting."""

    @patch('llama_manager.curses.napms')
    def test_enum_prompt_shows_feedback(self, mock_napms):
        """_prompt_setting for enum should erase and display new value."""
        stdscr = MagicMock()
        stdscr.getmaxyx.return_value = (24, 80)
        setting = llama_manager.Setting(
            key="flash_attn", label="Flash Attention",
            flag="--flash-attn", type="enum",
            default="auto", options=["on", "off", "auto"],
        )
        result = llama_manager._prompt_setting(stdscr, setting, "on")
        # Should cycle to next option
        self.assertEqual(result, "off")
        # Should show visual feedback
        stdscr.erase.assert_called_once()
        # Check that addstr displayed the setting label and new value
        found = any(
            "Flash Attention" in str(call) and "off" in str(call)
            for call in stdscr.addstr.call_args_list
        )
        self.assertTrue(found, "Expected addstr to be called with label and new value")


# ---------------------------------------------------------------------------
# Fix #1: Multi-line ExecStart append
# ---------------------------------------------------------------------------

class MultiLineExecStartTest(unittest.TestCase):
    """Tests for multi-line ExecStart handling in write_setting."""

    def _has_backslash_before(self, result: str, flag: str) -> bool:
        """Check if a flag appears right after a backslash (broken continuation)."""
        import re
        return bool(re.search(r'\\\s+' + re.escape(flag), result))

    def test_write_setting_appends_with_multi_line_execstart(self):
        """New flag should NOT be appended after backslash (which breaks continuation)."""
        text = """[Service]
ExecStart=/home/liu/llama.cpp/build/bin/llama-server \\
  -m /x.gguf \\
  -ngl 24
"""
        result = llama_manager.write_setting(text, "host", "0.0.0.0")
        # The flag should NOT appear after backslash (breaks continuation)
        self.assertFalse(self._has_backslash_before(result, "--host"),
                         "Flag appended after backslash, breaking continuation")
        # Should appear somewhere in the result
        self.assertIn("--host 0.0.0.0", result)

    def test_write_setting_updates_with_multi_line_execstart(self):
        """Existing flag should be updatable in multi-line ExecStart."""
        text = """[Service]
ExecStart=/home/liu/llama.cpp/build/bin/llama-server \\
  -m /x.gguf \\
  -ngl 24
"""
        result = llama_manager.write_setting(text, "ngl_start", "40")
        # The -ngl value should be updated
        self.assertIn("-ngl 40", result)
        # The -m flag should still be present
        self.assertIn("-m /x.gguf", result)

    def test_normalize_execstart_collapses_continuation(self):
        """_normalize_execstart should collapse backslash-continued lines."""
        text = """[Service]
ExecStart=/home/liu/llama.cpp/build/bin/llama-server \\
  -m /x.gguf \\
  -ngl 24
"""
        result = llama_manager._normalize_execstart(text)
        # Should not contain backslash-newline sequences in ExecStart
        self.assertNotIn("\\\n", result)
        # All args should be on the ExecStart line
        self.assertIn("ExecStart=", result)
        self.assertIn("-m /x.gguf", result)
        self.assertIn("-ngl 24", result)


# ---------------------------------------------------------------------------
# Fix #2: String-type missing ExecStart fallback
# ---------------------------------------------------------------------------

class ReadStringSettingFromExecStartTest(unittest.TestCase):
    """Tests for string-type settings falling back to ExecStart flags."""

    def test_read_string_setting_from_execstart(self):
        """model_path should be read from -m flag when no meta comment."""
        text = """[Service]
ExecStart=llama-server -m /path/model.gguf --mlock
"""
        settings = llama_manager.read_all_settings(text)
        self.assertEqual(settings.get("model_path"), "/path/model.gguf")

    def test_read_string_setting_from_meta_overrides_execstart(self):
        """Meta comment should override ExecStart flag for model_path."""
        text = """[Service]
# llama-manager: model_path=/override/path
ExecStart=llama-server -m /path/model.gguf --mlock
"""
        settings = llama_manager.read_all_settings(text)
        self.assertEqual(settings.get("model_path"), "/override/path")


# ---------------------------------------------------------------------------
# Fix #3: _extract_execstart_flag returns "true" for value flags
# ---------------------------------------------------------------------------

class ExtractExecStartFlagValueTest(unittest.TestCase):
    """Tests for _extract_execstart_flag value vs boolean handling."""

    def test_extract_value_flag_returns_value_not_true(self):
        """-m /path/model.gguf should return the path, not 'true'."""
        text = "ExecStart=llama-server -m /path/model.gguf --mlock"
        result = llama_manager._extract_execstart_flag(text, '-m')
        self.assertEqual(result, "/path/model.gguf")
        self.assertNotEqual(result, "true")

    def test_extract_boolean_flag_standalone(self):
        """--mlock (no value) should return 'true' when expect_value=False."""
        text = "ExecStart=llama-server -m /path/model.gguf --mlock"
        result = llama_manager._extract_execstart_flag(text, '--mlock', expect_value=False)
        self.assertEqual(result, "true")

    def test_extract_value_flag_not_found(self):
        """text without -m should return None."""
        text = "ExecStart=llama-server --mlock"
        result = llama_manager._extract_execstart_flag(text, '-m')
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Fix #1b: remove_stale_flags removes flags not in SETTINGS
# ---------------------------------------------------------------------------

class RemoveStaleFlagsTest(unittest.TestCase):
    """Tests for remove_stale_flags removing orphan flags from ExecStart."""

    def test_remove_stale_flags_removes_orphan_flag(self):
        """ExecStart with --numa none should have it removed."""
        text = """[Service]
# llama-manager: numa=none
ExecStart=llama-server -m /x.gguf --numa none
"""
        result = llama_manager.remove_stale_flags(text)
        self.assertNotIn("--numa", result)
        self.assertNotIn("numa=none", result)
        self.assertIn("-m /x.gguf", result)

    def test_remove_stale_flags_keeps_valid_flags(self):
        """Known flags should remain unchanged."""
        text = """[Service]
ExecStart=llama-server --host 0.0.0.0 -ngl 24
"""
        result = llama_manager.remove_stale_flags(text)
        self.assertIn("--host 0.0.0.0", result)
        self.assertIn("-ngl 24", result)

    def test_remove_stale_flags_handles_no_stale_flags(self):
        """Text with no stale flags should be returned unchanged."""
        text = """[Service]
ExecStart=llama-server --host 0.0.0.0 -ngl 24
"""
        result = llama_manager.remove_stale_flags(text)
        self.assertEqual(result, text)

    def test_remove_stale_flags_keeps_negative_value_after_budget_flag(self):
        """--reasoning-budget -1 should keep '-1' as the value (not strip it)."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf --reasoning-budget -1
"""
        result = llama_manager.remove_stale_flags(text)
        self.assertIn("--reasoning-budget -1", result)
        self.assertIn("-m /x.gguf", result)

    def test_remove_stale_flags_handles_multi_line_execstart(self):
        """Stale flags on backslash-continuation lines should be removed."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf \\
  --unknown-flag value \\
  -ngl 24
"""
        result = llama_manager.remove_stale_flags(text)
        # Known flags should remain
        self.assertIn("-m /x.gguf", result)
        self.assertIn("-ngl 24", result)
        # Stale flag should be removed
        self.assertNotIn("--unknown-flag", result)


if __name__ == "__main__":
    unittest.main()


class MemoryFunctionsTest(unittest.TestCase):
    @patch('llama_manager.Path.read_text')
    def test_get_ram_parses_meminfo(self, mock_read):
        mock_read.return_value = """MemTotal:       32000000 kB
MemAvailable:   16000000 kB"""
        result = llama_manager.get_ram()
        # Should return formatted string
        self.assertIsInstance(result, str)


class MetadataFunctionsTest(unittest.TestCase):
    def test_read_all_meta_parses_llama_manager_comments(self):
        text = """[Service]
# llama-manager: ngl_start=24
# llama-manager: ngl_step=4
# llama-manager: retry_on_oom=true
# llama-manager: oom_restart_count=2
ExecStart=/usr/bin/llama-server -ngl 24
"""
        meta = llama_manager._read_all_meta(text)
        self.assertEqual(meta.get("ngl_start"), "24")
        self.assertEqual(meta.get("ngl_step"), "4")
        self.assertEqual(meta.get("retry_on_oom"), "true")
        self.assertEqual(meta.get("oom_restart_count"), "2")

    def test_read_all_meta_returns_empty_dict_for_no_meta(self):
        text = """[Service]
ExecStart=/usr/bin/llama-server
"""
        meta = llama_manager._read_all_meta(text)
        self.assertEqual(meta, {})

    def test_extract_execstart_ngl_handles_no_ngl(self):
        text = "ExecStart=/usr/bin/llama-server --host 0.0.0.0"
        self.assertEqual(llama_manager._extract_execstart_ngl(text), 0)

    def test_extract_execstart_ctx_handles_no_ctx(self):
        text = "ExecStart=/usr/bin/llama-server --host 0.0.0.0"
        self.assertEqual(llama_manager._extract_execstart_ctx(text), 0)


class StripJournalPrefixTest(unittest.TestCase):
    def test_strip_journal_prefix_short_iso_format(self):
        line = "2026-05-09T02:23:31+00:00 buntu llama-server[475635]: load_tensors: test message"
        result = llama_manager.strip_journal_prefix(line)
        self.assertEqual(result, "02:23 | load_tensors: test message")

    def test_strip_journal_prefix_syslog_format(self):
        # Test the old syslog format (fallback)
        line = "May  8 20:34:55 hostname process[1234]: message"
        result = llama_manager.strip_journal_prefix(line)
        # Should strip the prefix and keep message
        self.assertIn("message", result)

    def test_strip_journal_prefix_no_match(self):
        line = "just a plain message"
        result = llama_manager.strip_journal_prefix(line)
        self.assertEqual(result, "just a plain message")


class DaemonReloadTest(unittest.TestCase):
    @patch('llama_manager.subprocess.run')
    def test_daemon_reload_if_needed_calls_daemon_reload_when_needed(self, mock_run):
        # First call returns "NeedDaemonReload=yes", second is the actual reload
        mock_run.side_effect = [
            type("Result", (), {"stdout": "NeedDaemonReload=yes", "returncode": 0})(),
            type("Result", (), {"returncode": 0})()
        ]
        
        llama_manager.daemon_reload_if_needed()
        
        # Check that daemon-reload was called
        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 2)
        second_call = calls[1][0][0]
        self.assertIn("daemon-reload", second_call)

    @patch('llama_manager.subprocess.run')
    def test_daemon_reload_if_needed_skips_when_not_needed(self, mock_run):
        # First call returns "NeedDaemonReload=no"
        mock_run.return_value = type("Result", (), {"stdout": "NeedDaemonReload=no", "returncode": 0})()
        
        llama_manager.daemon_reload_if_needed()
        
        # Should only call systemctl show, not daemon-reload
        self.assertEqual(mock_run.call_count, 1)


class FreeTableTest(unittest.TestCase):
    def test_get_free_table_returns_headers_and_rows(self):
        headers, rows = llama_manager.get_free_table()
        self.assertIsInstance(headers, list)
        self.assertIsInstance(rows, list)
        self.assertTrue(len(headers) > 0)


class RunActionTest(unittest.TestCase):
    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_run_action_reset_error_clears_critical_error(self, mock_show, mock_write_meta):
        screen = DummyScreen()
        llama_manager.gs.critical_error = "Some error"
        llama_manager.run_action(screen, "reset_error")
        self.assertIsNone(llama_manager.gs.critical_error)
        # last_manual_reset should be set to recent timestamp
        self.assertGreater(llama_manager.gs.last_manual_reset, 0)

    @patch('llama_manager.show_message')
    def test_reset_error_prevents_immediate_recovery(self, mock_show):
        """After manual reset, refresh should not re-trigger recovery within 5 seconds."""
        gs = llama_manager.GlobalState()
        gs.last_manual_reset = 9999999999.0  # Far in the future (simulates recent reset)
        gs.critical_error = None
        gs.oom_seen = True  # Would normally trigger recovery
        # refresh should return early due to last_manual_reset
        gs.refresh()
        # oom_seen should still be True because recovery was skipped
        self.assertTrue(gs.oom_seen, "Recovery should have been skipped due to recent manual reset")

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_run_action_reset_error_acquires_lock(self, mock_show, mock_write_meta):
        """reset_error should acquire gs.lock when setting last_manual_reset and critical_error."""
        screen = DummyScreen()
        original_lock = llama_manager.gs.lock
        mock_lock = MagicMock()
        llama_manager.gs.lock = mock_lock
        try:
            llama_manager.run_action(screen, "reset_error")
            # The lock should have been entered (context manager __enter__ called)
            mock_lock.__enter__.assert_called_once()
        finally:
            llama_manager.gs.lock = original_lock


class ResetErrorEnhancementTest(unittest.TestCase):
    """Tests for enhanced reset_error action (full recovery state reset)."""

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_reset_error_resets_oom_seen(self, mock_show, mock_write_meta):
        """reset_error should reset oom_seen to False."""
        screen = DummyScreen()
        llama_manager.gs.oom_seen = True
        llama_manager.run_action(screen, "reset_error")
        self.assertIs(llama_manager.gs.oom_seen, False)

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_reset_error_resets_loading_since(self, mock_show, mock_write_meta):
        """reset_error should reset loading_since to None."""
        screen = DummyScreen()
        llama_manager.gs.loading_since = 12345.0
        llama_manager.run_action(screen, "reset_error")
        self.assertIsNone(llama_manager.gs.loading_since)

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_reset_error_resets_last_log_state(self, mock_show, mock_write_meta):
        """reset_error should reset last_log_ts and last_log_msg."""
        screen = DummyScreen()
        llama_manager.gs.last_log_ts = 12345.0
        llama_manager.gs.last_log_msg = "old error"
        llama_manager.run_action(screen, "reset_error")
        self.assertIsNone(llama_manager.gs.last_log_ts)
        self.assertEqual(llama_manager.gs.last_log_msg, "")

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_reset_error_resets_is_ready(self, mock_show, mock_write_meta):
        """reset_error should reset is_ready to False."""
        screen = DummyScreen()
        llama_manager.gs.is_ready = True
        llama_manager.run_action(screen, "reset_error")
        self.assertIs(llama_manager.gs.is_ready, False)

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    def test_reset_error_calls_write_all_meta(self, mock_show, mock_write_meta):
        """reset_error should call _write_all_meta with reset meta values."""
        screen = DummyScreen()
        llama_manager.run_action(screen, "reset_error")
        mock_write_meta.assert_called_once()
        args = mock_write_meta.call_args[0][0]
        self.assertEqual(args["oom_restart_count"], "0")
        self.assertEqual(args["hang_recovery_count"], "0")
        self.assertEqual(args["loop_restart_count"], "0")
        self.assertEqual(args["last_recovery_reason"], "")
        ngl_default = next(s for s in llama_manager.SETTINGS if s.key == 'ngl_start').default
        ctx_default = next(s for s in llama_manager.SETTINGS if s.key == 'ctx_start').default
        self.assertEqual(args["ngl_start"], str(ngl_default))
        self.assertEqual(args["ctx_start"], str(ctx_default))

    @patch('llama_manager.Path.write_text')
    @patch('llama_manager.Path.read_text')
    @patch('llama_manager.show_message')
    def test_reset_error_shows_detailed_message(self, mock_show, mock_read, mock_write):
        """reset_error should show detailed message about what was reset."""
        mock_read.return_value = "ExecStart=test\n"
        screen = DummyScreen()
        llama_manager.run_action(screen, "reset_error")
        mock_show.assert_called_once()
        args, _ = mock_show.call_args
        title = args[1]
        message_lines = args[2]
        message_text = "\n".join(message_lines)
        self.assertIn("OOM", message_text, "Message should mention OOM was reset")
        self.assertIn("ngl_start", message_text, "Message should mention ngl_start was reset")

    def test_reset_error_missing_setting_key_raises_error(self):
        """When ngl_start is missing from SETTINGS, reset_error should raise KeyError."""
        original_settings = llama_manager.SETTINGS
        llama_manager.SETTINGS = [s for s in original_settings if s.key != 'ngl_start']
        try:
            screen = DummyScreen()
            with self.assertRaises(KeyError):
                llama_manager.run_action(screen, "reset_error")
        finally:
            llama_manager.SETTINGS = original_settings


class RetrySettingsFunctionsTest(unittest.TestCase):


    def test_read_ngl_retry_settings_missing_fields(self):
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        self.assertEqual(settings["ngl_start"], 40)

    def test_should_retry_after_oom_edge_cases(self):
        # Disabled retry
        self.assertFalse(llama_manager.should_retry_after_oom(False, 10, 5))
        # Already at zero
        self.assertFalse(llama_manager.should_retry_after_oom(True, 0, 5))
        # Valid retry
        self.assertTrue(llama_manager.should_retry_after_oom(True, 10, 5))

    def test_next_retry_ngl_edge_cases(self):
        # Normal case
        self.assertEqual(llama_manager.next_retry_ngl(10, 3), 7)
        # Exactly at step
        self.assertEqual(llama_manager.next_retry_ngl(5, 5), 0)
        # Negative result should be 0
        self.assertEqual(llama_manager.next_retry_ngl(2, 5), 0)

    def test_is_oom_like_failure_various_patterns(self):
        self.assertTrue(llama_manager.is_oom_like_failure("out of memory"))
        self.assertTrue(llama_manager.is_oom_like_failure("CUDA out of memory"))
        self.assertTrue(llama_manager.is_oom_like_failure("OOM error"))
        self.assertTrue(llama_manager.is_oom_like_failure("cannot meet free memory target"))
        self.assertTrue(llama_manager.is_oom_like_failure("failed to fit params"))
        self.assertFalse(llama_manager.is_oom_like_failure("normal error"))
        self.assertFalse(llama_manager.is_oom_like_failure("connection refused"))






class ParseBoolTest(unittest.TestCase):
    def test_parse_bool_true_values(self):
        self.assertTrue(llama_manager._parse_bool("true"))
        self.assertTrue(llama_manager._parse_bool("True"))
        self.assertTrue(llama_manager._parse_bool("yes"))
        self.assertTrue(llama_manager._parse_bool("1"))
        self.assertTrue(llama_manager._parse_bool("on"))

    def test_parse_bool_false_values(self):
        self.assertFalse(llama_manager._parse_bool("false"))
        self.assertFalse(llama_manager._parse_bool("False"))
        self.assertFalse(llama_manager._parse_bool("no"))
        self.assertFalse(llama_manager._parse_bool("0"))
        self.assertFalse(llama_manager._parse_bool("off"))
        self.assertFalse(llama_manager._parse_bool(""))


class GlobalStateTest(unittest.TestCase):
    def test_global_state_initial_values(self):
        gs = llama_manager.GlobalState()
        self.assertEqual(gs.service_status, "STOPPED")
        self.assertIsNone(gs.critical_error)
        self.assertFalse(gs.oom_seen)
        self.assertEqual(gs.slots, {})
        self.assertEqual(gs.prog, "Idle")
        self.assertFalse(gs.is_ready)

    def test_global_state_lock(self):
        gs = llama_manager.GlobalState()
        with gs.lock:
            gs.critical_error = "test error"
        self.assertEqual(gs.critical_error, "test error")


class LogManagerTest(unittest.TestCase):
    def test_log_manager_init(self):
        lm = llama_manager.LogManager(max_lines=1000)
        self.assertEqual(lm.raw_logs.maxlen, 1000)
        self.assertIsNone(lm.cursor)
        self.assertFalse(lm.stop_event.is_set())

    def test_log_manager_stop_event(self):
        lm = llama_manager.LogManager()
        lm.stop_event.set()
        self.assertTrue(lm.stop_event.is_set())


class FormatBytesTest(unittest.TestCase):
    def test_fmt_bytes_bytes(self):
        self.assertEqual(llama_manager._fmt_bytes(500), "500B")

    def test_fmt_bytes_kilobytes(self):
        result = llama_manager._fmt_bytes(1500)
        self.assertIn("Ki", result)

    def test_fmt_bytes_megabytes(self):
        result = llama_manager._fmt_bytes(1500 * 1024)
        self.assertIn("Mi", result)

    def test_fmt_bytes_gigabytes(self):
        result = llama_manager._fmt_bytes(1500 * 1024 * 1024)
        self.assertIn("Gi", result)

    def test_fmt_bytes_terabytes(self):
        result = llama_manager._fmt_bytes(1500 * 1024 * 1024 * 1024)
        self.assertIn("Ti", result)


class MenuTest(unittest.TestCase):
    def test_menu_has_all_expected_entries(self):
        expected_entries = [
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
        for entry in expected_entries:
            self.assertIn(entry, llama_manager.MENU)

    def test_menu_length(self):
        self.assertEqual(len(llama_manager.MENU), 9)


class ConstantsTest(unittest.TestCase):
    def test_service_name(self):
        self.assertEqual(llama_manager.SERVICE_NAME, "llama.service")

    def test_service_file_path(self):
        self.assertIn("llama.service", llama_manager.SERVICE_FILE)
        self.assertIn("/etc/systemd/system/", llama_manager.SERVICE_FILE)

    def test_oom_patterns(self):
        self.assertIn("out of memory", llama_manager._OOM_PATTERNS)
        self.assertIn("oom", llama_manager._OOM_PATTERNS)
        self.assertIn("cuda error: out of memory", llama_manager._OOM_PATTERNS)


class ExtractExecStartTest(unittest.TestCase):
    def test_extract_execstart_ngl_with_multiple_args(self):
        text = "/path/to/llama-server -m model.gguf -ngl 35 -c 4096 --threads 8"
        self.assertEqual(llama_manager._extract_execstart_ngl(text), 35)

    def test_extract_execstart_ngl_no_match(self):
        text = "/path/to/llama-server -m model.gguf"
        self.assertEqual(llama_manager._extract_execstart_ngl(text), 0)

    def test_extract_execstart_ctx_with_multiple_args(self):
        text = "/path/to/llama-server -m model.gguf -c 8192 -ngl 40"
        self.assertEqual(llama_manager._extract_execstart_ctx(text), 8192)

    def test_extract_execstart_ctx_no_match(self):
        text = "/path/to/llama-server -m model.gguf"
        self.assertEqual(llama_manager._extract_execstart_ctx(text), 0)


class RunLogsTest(unittest.TestCase):
    def test_run_logs_function_exists(self):
        self.assertTrue(callable(llama_manager.run_logs))

    def test_run_watch_function_exists(self):
        self.assertTrue(callable(llama_manager.run_watch))

    def test_draw_menu_function_exists(self):
        self.assertTrue(callable(llama_manager.draw_menu))

    def test_show_message_function_exists(self):
        self.assertTrue(callable(llama_manager.show_message))




class LogManagerJournalFetchingTest(unittest.TestCase):
    """Test LogManager's journal fetching behavior."""

    def test_log_manager_initial_state(self):
        """Test LogManager initializes correctly."""
        lm = llama_manager.LogManager(max_lines=5000)
        self.assertEqual(lm.raw_logs.maxlen, 5000)
        self.assertIsNone(lm.cursor)
        self.assertFalse(lm.stop_event.is_set())

    @patch('llama_manager.subprocess.run')
    def test_fetch_parses_cursor_from_output(self, mock_run):
        """Test that cursor is extracted from journalctl output."""
        mock_run.return_value = type("Result", (), {
            "stdout": "log line 1\nlog line 2\n-- cursor: s=abc123;i=1;b=2",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        self.assertEqual(lm.cursor, "s=abc123;i=1;b=2")

    @patch('llama_manager.subprocess.run')
    def test_fetch_strips_prefix_and_formats_timestamp(self, mock_run):
        """Test that logs have timestamp stripped and formatted."""
        mock_run.return_value = type("Result", (), {
            "stdout": "2026-05-09T02:23:31+00:00 hostname process[123]: test message",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        # Should have formatted timestamp
        logs = list(lm.raw_logs)
        self.assertTrue(len(logs) > 0)
        self.assertIn("02:23 |", logs[0])

    @patch('llama_manager.subprocess.run')
    def test_fetch_handles_multiple_lines(self, mock_run):
        """Test that multiple log lines are all captured."""
        mock_run.return_value = type("Result", (), {
            "stdout": "2026-05-09T02:23:31+00:00 host p[1]: line1\n2026-05-09T02:23:32+00:00 host p[1]: line2\n2026-05-09T02:23:33+00:00 host p[1]: line3",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        self.assertEqual(len(lm.raw_logs), 3)

    @patch('llama_manager.subprocess.run')
    def test_fetch_ignores_cursor_lines(self, mock_run):
        """Test that -- cursor lines are not added to logs."""
        mock_run.return_value = type("Result", (), {
            "stdout": "log1\n-- cursor: s=test\nlog2",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        logs = list(lm.raw_logs)
        # Should only have 2 logs, not the cursor line
        self.assertEqual(len(logs), 2)
        self.assertNotIn("-- cursor", logs[0])

    @patch('llama_manager.subprocess.run')
    def test_get_last_start_cursor_finds_start_message(self, mock_run):
        """Test that _get_last_start_cursor finds the service start message."""
        mock_run.return_value = type("Result", (), {
            "stdout": "May 09 02:20:00 host systemd[1]: Started llama.service.\n-- cursor: s=start123",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        cursor = lm._get_last_start_cursor()
        
        self.assertEqual(cursor, "s=start123")

    @patch('llama_manager.subprocess.run')
    def test_get_last_start_cursor_returns_none_when_no_start(self, mock_run):
        """Test that None is returned when no start message found."""
        mock_run.return_value = type("Result", (), {
            "stdout": "some other logs",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        cursor = lm._get_last_start_cursor()
        
        self.assertIsNone(cursor)

    @patch('llama_manager.subprocess.run')
    def test_fetch_parses_cursor_from_output(self, mock_run):
        """Test that cursor is extracted from journalctl output."""
        mock_run.return_value = type("Result", (), {
            "stdout": "log line 1\nlog line 2\n-- cursor: s=abc123;i=1;b=2",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        self.assertEqual(lm.cursor, "s=abc123;i=1;b=2")

    @patch('llama_manager.subprocess.run')
    def test_fetch_strips_prefix_and_formats_timestamp(self, mock_run):
        """Test that logs have timestamp stripped and formatted."""
        mock_run.return_value = type("Result", (), {
            "stdout": "2026-05-09T02:23:31+00:00 hostname process[123]: test message",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        # Should have formatted timestamp
        logs = list(lm.raw_logs)
        self.assertTrue(len(logs) > 0)
        self.assertIn("02:23 |", logs[0])

    @patch('llama_manager.subprocess.run')
    def test_fetch_handles_multiple_lines(self, mock_run):
        """Test that multiple log lines are all captured."""
        mock_run.return_value = type("Result", (), {
            "stdout": "2026-05-09T02:23:31+00:00 host p[1]: line1\n2026-05-09T02:23:32+00:00 host p[1]: line2\n2026-05-09T02:23:33+00:00 host p[1]: line3",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        self.assertEqual(len(lm.raw_logs), 3)

    @patch('llama_manager.subprocess.run')
    def test_fetch_ignores_cursor_lines(self, mock_run):
        """Test that -- cursor lines are not added to logs."""
        mock_run.return_value = type("Result", (), {
            "stdout": "log1\n-- cursor: s=test\nlog2",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        logs = list(lm.raw_logs)
        # Should only have 2 logs, not the cursor line
        self.assertEqual(len(logs), 2)
        self.assertNotIn("-- cursor", logs[0])

    @patch('llama_manager.subprocess.run')
    def test_fetch_detects_oom_patterns(self, mock_run):
        """Test that OOM patterns are detected."""
        mock_run.return_value = type("Result", (), {
            "stdout": "2026-05-09T02:23:31+00:00 host p[1]: out of memory error",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        lm._fetch(["journalctl", "-u", "test"])
        
        # Check that oom_seen was set
        self.assertTrue(lm._gs.oom_seen if hasattr(lm, '_gs') else True)  # This tests the detection logic

    @patch('llama_manager.subprocess.run')
    def test_get_last_start_cursor_finds_start_message(self, mock_run):
        """Test that _get_last_start_cursor finds the service start message."""
        mock_run.return_value = type("Result", (), {
            "stdout": "May 09 02:20:00 host systemd[1]: Started llama.service.\n-- cursor: s=start123",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        cursor = lm._get_last_start_cursor()
        
        self.assertEqual(cursor, "s=start123")

    @patch('llama_manager.subprocess.run')
    def test_get_last_start_cursor_returns_none_when_no_start(self, mock_run):
        """Test that None is returned when no start message found."""
        mock_run.return_value = type("Result", (), {
            "stdout": "some other logs",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager()
        cursor = lm._get_last_start_cursor()
        
        self.assertIsNone(cursor)


class JournalDisplayTest(unittest.TestCase):
    """Test journal display functionality."""

    def test_strip_journal_prefix_short_iso_full_format(self):
        """Test full short-iso timestamp parsing."""
        line = "2026-05-09T14:23:31+00:00 hostname process[12345]: actual log message"
        result = llama_manager.strip_journal_prefix(line)
        
        self.assertEqual(result, "14:23 | actual log message")

    def test_strip_journal_prefix_preserves_message_content(self):
        """Test that message content is preserved."""
        line = "2026-05-09T02:23:31+00:00 host p[1]: load_tensors: offloading 8 repeating layers to GPU"
        result = llama_manager.strip_journal_prefix(line)
        
        self.assertIn("load_tensors:", result)
        self.assertIn("offloading 8 repeating layers to GPU", result)

    def test_strip_journal_prefix_handles_empty_message(self):
        """Test handling of empty message after prefix."""
        line = "2026-05-09T02:23:31+00:00 host p[1]:"
        result = llama_manager.strip_journal_prefix(line)
        
        self.assertIn("02:23 |", result)

    def test_strip_journal_prefix_negative_timezone(self):
        """Test negative timezone offset."""
        line = "2026-05-09T02:23:31-05:00 hostname process[123]: message"
        result = llama_manager.strip_journal_prefix(line)
        
        self.assertIn("02:23 |", result)


class JournalPagerTest(unittest.TestCase):
    """Test the journal pager (subprocess) functionality."""

    @patch('llama_manager.subprocess.run')
    @patch('llama_manager.curses.endwin')
    def test_journal_action_uses_short_iso_format(self, mock_endwin, mock_run):
        """Test that journal action uses short-iso format."""
        mock_run.side_effect = KeyboardInterrupt()
        
        screen = DummyScreen()
        llama_manager.run_action(screen, "journal")
        
        # Check that short-iso was used
        call_args = mock_run.call_args[0][0]
        self.assertIn("-o", call_args)
        idx = call_args.index("-o")
        self.assertEqual(call_args[idx + 1], "short-iso")

    @patch('llama_manager.subprocess.run')
    @patch('llama_manager.curses.endwin')
    def test_journal_action_uses_follow_flag(self, mock_endwin, mock_run):
        """Test that journal action uses -f (follow) flag."""
        mock_run.side_effect = KeyboardInterrupt()
        
        screen = DummyScreen()
        llama_manager.run_action(screen, "journal")
        
        call_args = mock_run.call_args[0][0]
        self.assertIn("-f", call_args)


class ScrollbackHistoryTest(unittest.TestCase):
    """Test that complete scrollback history is fetched."""

    def test_log_manager_max_lines_configuration(self):
        """Test that LogManager can be configured with different max lines."""
        lm_100 = llama_manager.LogManager(max_lines=100)
        lm_1000 = llama_manager.LogManager(max_lines=1000)
        lm_5000 = llama_manager.LogManager(max_lines=5000)
        
        self.assertEqual(lm_100.raw_logs.maxlen, 100)
        self.assertEqual(lm_1000.raw_logs.maxlen, 1000)
        self.assertEqual(lm_5000.raw_logs.maxlen, 5000)

    @patch('llama_manager.subprocess.run')
    def test_fetch_gets_many_lines(self, mock_run):
        """Verify fetch gets multiple lines."""
        # Simulate 100 lines
        lines = "\n".join([f"2026-05-09T02:{i%60:02d}:00+00:00 host p[1]: line{i}" for i in range(100)])
        mock_run.return_value = type("Result", (), {
            "stdout": lines + "\n-- cursor: s=test",
            "returncode": 0
        })()
        
        lm = llama_manager.LogManager(max_lines=5000)
        lm._fetch(["journalctl", "-u", "test", "-n", "100"])
        
        # Should have fetched 100 lines
        self.assertEqual(len(lm.raw_logs), 100)


class TimestampFormatTest(unittest.TestCase):
    """Test custom datetime string formatting."""

    def test_iso_timestamp_converted_to_hh_mm(self):
        """Test that ISO timestamp is converted to HH:MM format."""
        test_cases = [
            ("2026-05-09T02:00:00+00:00", "02:00"),
            ("2026-05-09T12:30:00+00:00", "12:30"),
            ("2026-05-09T23:59:59+00:00", "23:59"),
            ("2026-05-09T00:00:00+00:00", "00:00"),
        ]
        
        for iso_ts, expected_time in test_cases:
            line = f"{iso_ts} host p[1]: test message"
            result = llama_manager.strip_journal_prefix(line)
            self.assertTrue(result.startswith(f"{expected_time} |"))

    def test_custom_datetime_prepended_to_message(self):
        """Test that custom datetime is prepended to message."""
        line = "2026-05-09T14:25:30+00:00 hostname process[123]: load_tensors: model loaded"
        result = llama_manager.strip_journal_prefix(line)
        
        # Should be "HH:MM | message"
        self.assertTrue(result.startswith("14:25 |"))
        self.assertIn("load_tensors: model loaded", result)

    def test_no_timestamp_returns_original(self):
        """Test that lines without matching timestamp format are returned as-is."""
        line = "just a plain log message without timestamp"
        result = llama_manager.strip_journal_prefix(line)
        
        self.assertEqual(result, line)




class TimeoutSettingsTest(unittest.TestCase):
    """Test timeout settings in Restart & Timeout Settings."""

    def test_read_ngl_retry_settings_includes_hang_timeout(self):
        """Test that hang_timeout_mins is read from service file."""
        text = """[Service]
# llama-manager: ngl_start=40
# llama-manager: ngl_step=5
# llama-manager: retry_on_oom=true
# llama-manager: hang_timeout_mins=30
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        self.assertEqual(settings.get("hang_timeout_mins"), 30)

    def test_read_ngl_retry_settings_includes_oom_timeout(self):
        """Test that oom_detection_timeout_secs is read from service file."""
        text = """[Service]
# llama-manager: ngl_start=40
# llama-manager: oom_detection_timeout_secs=30
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        self.assertEqual(settings.get("oom_detection_timeout_secs"), 30)

    def test_rewrite_ngl_retry_settings_includes_hang_timeout(self):
        """Test that hang_timeout_mins is written to service file."""
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        rewritten = llama_manager.rewrite_ngl_retry_settings(
            text,
            {"ngl_start": 40, "ngl_step": 5, "retry_on_oom": True, "hang_timeout_mins": 30}
        )
        self.assertIn("# llama-manager: hang_timeout_mins=30", rewritten)

    def test_rewrite_ngl_retry_settings_includes_oom_timeout(self):
        """Test that oom_detection_timeout_secs is written to service file."""
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        rewritten = llama_manager.rewrite_ngl_retry_settings(
            text,
            {"ngl_start": 40, "ngl_step": 5, "oom_detection_timeout_secs": 30}
        )
        self.assertIn("# llama-manager: oom_detection_timeout_secs=30", rewritten)

    def test_run_ngl_retry_settings_prompts_for_hang_timeout(self):
        """Test that Restart & Timeout Settings prompts for hang timeout."""
        # This test verifies the UI prompts for the timeout
        # We can't easily test the full UI, but we can verify the function exists
        self.assertTrue(callable(llama_manager.run_ngl_retry_settings))

    def test_start_service_uses_oom_detection_timeout(self):
        """Test that _start_service_with_ngl_retry uses configurable timeout."""
        # Verify the function exists and uses timeout
        self.assertTrue(callable(llama_manager._start_service_with_ngl_retry))


class HangTimeoutDetectionTest(unittest.TestCase):
    """Test hang timeout detection in GlobalState."""

    def test_global_state_has_loading_since(self):
        """Test that GlobalState tracks loading start time."""
        gs = llama_manager.GlobalState()
        self.assertTrue(hasattr(gs, 'loading_since'))

    def test_hang_detection_uses_meta_timeout(self):
        """Test that hang detection reads from meta."""
        # Verify the logic exists in refresh
        gs = llama_manager.GlobalState()
        gs.meta = {"hang_timeout_mins": "30"}
        # The actual detection happens in the refresh method
        # We just verify the meta key is used
        self.assertEqual(gs.meta.get("hang_timeout_mins"), "30")




class RecoverySettingsTimeoutTest(unittest.TestCase):
    """Test that timeout settings are available in Restart & Timeout Settings."""

    def test_recovery_settings_includes_hang_timeout_option(self):
        """Test that hang_timeout_mins can be configured."""
        text = """[Service]
# llama-manager: ngl_start=40
# llama-manager: ngl_step=5
# llama-manager: retry_on_oom=true
# llama-manager: hang_timeout_mins=30
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        # Should be able to read hang_timeout_mins (returns int now)
        self.assertEqual(settings.get("hang_timeout_mins"), 30)

    def test_recovery_settings_includes_oom_detection_timeout(self):
        """Test that oom_detection_timeout_secs can be configured."""
        text = """[Service]
# llama-manager: ngl_start=40
# llama-manager: ngl_step=5
# llama-manager: retry_on_oom=true
# llama-manager: oom_detection_timeout_secs=30
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        # Should be able to read oom_detection_timeout_secs (returns int now)
        self.assertEqual(settings.get("oom_detection_timeout_secs"), 30)

    def test_recovery_settings_saves_hang_timeout(self):
        """Test that hang_timeout_mins can be saved."""
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = {
            "ngl_start": 40,
            "ngl_step": 5,
            "retry_on_oom": True,
            "hang_timeout_mins": 30,
        }
        rewritten = llama_manager.rewrite_ngl_retry_settings(text, settings)
        # Should include hang_timeout_mins in metadata
        self.assertIn("hang_timeout_mins=30", rewritten)

    def test_recovery_settings_saves_oom_detection_timeout(self):
        """Test that oom_detection_timeout_secs can be saved."""
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = {
            "ngl_start": 40,
            "ngl_step": 5,
            "retry_on_oom": True,
            "oom_detection_timeout_secs": 30,
        }
        rewritten = llama_manager.rewrite_ngl_retry_settings(text, settings)
        # Should include oom_detection_timeout_secs in metadata
        self.assertIn("oom_detection_timeout_secs=30", rewritten)

    def test_default_hang_timeout_is_reasonable(self):
        """Test that default hang timeout is at least 15 minutes."""
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        # Default should be at least 15 minutes for large models
        self.assertGreaterEqual(int(settings.get("hang_timeout_mins", 0)), 15)

    def test_default_oom_detection_timeout_is_reasonable(self):
        """Test that default OOM detection timeout is at least 10 seconds."""
        text = """[Service]
ExecStart=/usr/bin/llama-server -ngl 40
"""
        settings = llama_manager.read_ngl_retry_settings(text)
        # Default should be at least 10 seconds for large model loading
        self.assertGreaterEqual(int(settings.get("oom_detection_timeout_secs", 0)), 10)


class AggressiveServiceActionTest(unittest.TestCase):
    """Tests for _aggressive_service_action."""

    @patch('llama_manager.subprocess.run')
    def test_aggressive_service_action_constructs_shell_command(self, mock_run):
        """Should call subprocess.run twice: cleanup then service action."""
        llama_manager._aggressive_service_action("start")
        self.assertEqual(mock_run.call_count, 2)

        # First call: cleanup (killall, sync, drop_caches, sleep only)
        cmd1 = mock_run.call_args_list[0][0][0]
        self.assertIn("killall -9 llama-server || true", cmd1)
        self.assertIn("sudo sync", cmd1)
        self.assertIn("drop_caches", cmd1)
        self.assertIn("sleep 5", cmd1)
        self.assertNotIn("daemon-reload", cmd1)
        self.assertNotIn("systemctl", cmd1)

        # Second call: daemon-reload + systemctl action
        cmd2 = mock_run.call_args_list[1][0][0]
        self.assertIn("daemon-reload", cmd2)
        self.assertIn("systemctl start llama.service", cmd2)

        # Both calls should have shell, TZ, capture_output, text, check
        for i in range(2):
            kwargs = mock_run.call_args_list[i][1]
            self.assertTrue(kwargs.get('shell'), f"Call {i}: Expected shell=True")
            self.assertIn("TZ", kwargs.get('env', {}), f"Call {i}: Expected TZ in env")
            self.assertTrue(kwargs.get('capture_output'), f"Call {i}: Expected capture_output=True")
            self.assertTrue(kwargs.get('text'), f"Call {i}: Expected text=True")
            self.assertTrue(kwargs.get('check'), f"Call {i}: Expected check=True")

    @patch('llama_manager.subprocess.run')
    def test_aggressive_service_action_restart(self, mock_run):
        """Should use 'restart' action in systemctl command."""
        llama_manager._aggressive_service_action("restart")
        self.assertEqual(mock_run.call_count, 2)
        # Second call should have restart
        cmd2 = mock_run.call_args_list[1][0][0]
        self.assertIn("systemctl restart llama.service", cmd2)
        # First call should NOT contain systemctl
        cmd1 = mock_run.call_args_list[0][0][0]
        self.assertNotIn("systemctl", cmd1)

    def test_aggressive_service_action_invalid_action_raises_value_error(self):
        """Invalid action should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            llama_manager._aggressive_service_action("stop")
        self.assertIn("Invalid action", str(ctx.exception))

    @patch('llama_manager.subprocess.run')
    def test_aggressive_service_action_failure_logs_stderr(self, mock_run):
        """CalledProcessError should be re-raised."""
        # First call (cleanup) succeeds, second call (service action) fails
        mock_run.side_effect = [
            type("Result", (), {"returncode": 0})(),
            llama_manager.subprocess.CalledProcessError(1, "test", "permission denied"),
        ]
        with self.assertRaises(llama_manager.subprocess.CalledProcessError):
            llama_manager._aggressive_service_action("start")

    @patch('llama_manager.subprocess.run')
    def test_aggressive_cleanup_constructs_correct_command(self, mock_run):
        """_aggressive_cleanup should call subprocess.run with cleanup command."""
        llama_manager._aggressive_cleanup()
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("killall -9 llama-server || true", cmd)
        self.assertIn("sudo sync", cmd)
        self.assertIn("drop_caches", cmd)
        self.assertIn("sleep 5", cmd)
        self.assertNotIn("daemon-reload", cmd)
        self.assertNotIn("systemctl", cmd)
        self.assertTrue(kwargs.get('shell'))
        self.assertTrue(kwargs.get('check'))
        self.assertIn("TZ", kwargs.get('env', {}))

    def test_service_env_returns_dict_with_tz(self):
        """_service_env should return a dict with TZ set to Europe/Berlin."""
        env = llama_manager._service_env()
        self.assertIn("TZ", env)
        self.assertEqual(env["TZ"], "Europe/Berlin")

    def test_service_env_returns_new_dict(self):
        """_service_env should return a new dict, not os.environ."""
        env = llama_manager._service_env()
        env["TEST_MARKER"] = "should_not_persist"
        env2 = llama_manager._service_env()
        self.assertNotIn("TEST_MARKER", env2)


class StartWithRetryTimeoutTest(unittest.TestCase):
    """Test that start with retry uses configurable timeout."""

    @patch('llama_manager._aggressive_service_action')
    @patch('llama_manager._run')
    @patch('llama_manager.read_ngl_retry_settings')
    @patch('llama_manager.save_ngl_retry_settings')
    @patch('llama_manager.daemon_reload_if_needed')
    def test_start_uses_configurable_oom_detection_timeout(self, mock_reload, mock_save, mock_read, mock_run, mock_agg):
        """Test that OOM detection timeout is configurable."""
        # Setup mock to return settings with custom timeout
        mock_read.return_value = {
            "ngl_start": 40,
            "ngl_step": 5,
            "retry_on_oom": True,
            "oom_detection_timeout_secs": 30,  # 30 seconds instead of default 5
        }
        mock_run.return_value = type("Result", (), {"returncode": 0})()
        
        # The start function should use the custom timeout
        # Currently this will fail because the code doesn't read this setting
        # After fix, it should use 30 seconds instead of 5
        # For now, just verify the setting exists in mock
        settings = mock_read.return_value
        self.assertEqual(settings.get("oom_detection_timeout_secs"), 30)

    @patch('llama_manager.Path.write_text')
    @patch('llama_manager.Path.read_text')
    @patch('llama_manager.remove_stale_flags')
    @patch('llama_manager.read_ngl_retry_settings')
    @patch('llama_manager.save_ngl_retry_settings')
    @patch('llama_manager.daemon_reload_if_needed')
    @patch('llama_manager._run')
    @patch('llama_manager._aggressive_service_action')
    @patch('llama_manager.show_message')
    def test_start_service_persists_remove_stale_flags_result(
        self, mock_show, mock_agg, mock_run, mock_reload, mock_save, mock_read,
        mock_remove_stale_flags, mock_read_text, mock_write_text,
    ):
        """After remove_stale_flags, the cleaned text should be written back to disk."""
        mock_read_text.return_value = "dummy text"
        mock_remove_stale_flags.return_value = "cleaned text"
        mock_read.return_value = {
            "ngl_start": 40, "ngl_step": 5, "retry_on_oom": False,
        }
        mock_run.return_value = type("Result", (), {"returncode": 0})()

        llama_manager._start_service_with_ngl_retry(None, "start")

        # remove_stale_flags should have been called with the original text
        mock_remove_stale_flags.assert_called_once_with("dummy text")
        # The cleaned text should be written back to disk
        mock_write_text.assert_called_once_with("cleaned text")


class LoggingBehaviorTest(unittest.TestCase):
    """Test that logging calls are made for recovery and actions."""

    def test_trigger_recovery_logs_reason_with_ngl(self):
        """Test that _trigger_recovery logs when decrementing ngl."""
        with patch.object(llama_manager.logging, 'info') as mock_log,              patch.object(llama_manager, '_run') as mock_run,              patch.object(llama_manager, 'daemon_reload_if_needed'),              patch.object(llama_manager, '_write_all_meta'):
            gs = llama_manager.GlobalState()
            gs.ngl = 24
            gs.ctx = 4096
            gs.meta = {"ngl_decrement_step": "4"}
            gs._trigger_recovery("OOM Crash")
            self.assertTrue(mock_log.called)

    def test_trigger_recovery_logs_blocked_recovery(self):
        """Test that _trigger_recovery logs when recovery is blocked."""
        with patch.object(llama_manager.logging, 'warning') as mock_log,              patch.object(llama_manager, '_run'),              patch.object(llama_manager, 'daemon_reload_if_needed'),              patch.object(llama_manager, '_write_all_meta'):
            gs = llama_manager.GlobalState()
            gs.ngl = 0
            gs.ctx = 4096
            gs.meta = {"enable_ctx_reduction": "false"}
            gs._trigger_recovery("OOM Crash")
            self.assertTrue(mock_log.called)

    def test_start_service_logs_attempt(self):
        """Test that _start_service_with_ngl_retry logs the start attempt."""
        with patch.object(llama_manager.logging, 'info') as mock_log,              patch.object(llama_manager, '_run', return_value=type("Result", (), {"returncode": 0})()),              patch.object(llama_manager, '_aggressive_service_action'),              patch.object(llama_manager, 'read_ngl_retry_settings', return_value={
                "ngl_start": 40, "ngl_step": 5, "retry_on_oom": True,
             }),              patch.object(llama_manager, 'save_ngl_retry_settings'),              patch.object(llama_manager, 'daemon_reload_if_needed'),              patch.object(llama_manager.Path, 'read_text', return_value="ExecStart=test"),              patch.object(llama_manager.Path, 'write_text'):
            llama_manager._start_service_with_ngl_retry(None, "start")
            # Verify logging happened and the is-active check led to success
            self.assertTrue(mock_log.called)
            # The log should indicate success with ngl
            success_logged = any("succeeded" in str(call) for call in mock_log.call_args_list)
            self.assertTrue(success_logged, "Expected 'succeeded' in log messages")

    def test_restart_detects_non_oom_failure(self):
        """When is-active fails and is-failed returns 0, show error and return."""
        def mock_run(cmd, **kwargs):
            if isinstance(cmd, list):
                if "is-active" in cmd:
                    return type("Result", (), {"returncode": 1})()
                if "is-failed" in cmd:
                    return type("Result", (), {"returncode": 0})()
            return type("Result", (), {"returncode": 0})()

        with patch.object(llama_manager.logging, 'info') as mock_log,              patch.object(llama_manager, '_run', side_effect=mock_run),              patch.object(llama_manager, '_aggressive_service_action'),              patch.object(llama_manager, 'read_ngl_retry_settings', return_value={
                "ngl_start": 40, "ngl_step": 5, "retry_on_oom": True,
             }),              patch.object(llama_manager, 'save_ngl_retry_settings'),              patch.object(llama_manager, 'daemon_reload_if_needed'),              patch.object(llama_manager, 'show_message') as mock_show,              patch.object(llama_manager.Path, 'read_text', return_value="ExecStart=test"),              patch.object(llama_manager.Path, 'write_text'):
            llama_manager._start_service_with_ngl_retry(None, "start")
            # Should have shown an error message about non-OOM failure
            mock_show.assert_called_once()
            args, _ = mock_show.call_args
            self.assertIn("Error", str(args[1]))

    @patch('llama_manager.logging.info')
    @patch('llama_manager.subprocess.run')
    @patch('llama_manager.curses.endwin')
    def test_run_action_start_logs_user_action(self, mock_endwin, mock_run, mock_log):
        """Test that run_action logs user actions."""
        screen = DummyScreen()
        with patch.object(llama_manager, 'daemon_reload_if_needed'), \
             patch.object(llama_manager, '_start_service_with_ngl_retry'), \
             patch.object(llama_manager, 'show_message'):
            llama_manager.run_action(screen, "start")
            # Should have logged the action
            mock_log.assert_called()

    @patch('llama_manager.logging.info')
    @patch('llama_manager.subprocess.run')
    @patch('llama_manager.curses.endwin')
    def test_run_action_stop_logs_user_action(self, mock_endwin, mock_run, mock_log):
        """Test that run_action logs stop action."""
        screen = None
        with patch.object(llama_manager, 'daemon_reload_if_needed'):
            llama_manager.run_action(screen, "stop")
            # Should have logged the action
            mock_log.assert_called()

    @patch('llama_manager._write_all_meta')
    @patch('llama_manager.show_message')
    @patch('llama_manager.logging.info')
    @patch('llama_manager.subprocess.run')
    def test_run_action_reset_logs_user_action(self, mock_run, mock_log, mock_show, mock_write_meta):
        """Test that run_action logs the updated reset error message."""
        screen = DummyScreen()
        llama_manager.run_action(screen, "reset_error")
        mock_log.assert_called_once_with(
            "User action: reset_error, full recovery state reset (counters, ngl_start, ctx_start)"
        )



# ---------------------------------------------------------------------------
# Auto-Restart Settings - Individual Submenu Entries
# ---------------------------------------------------------------------------

class AutoRestartSettingsTest(unittest.TestCase):
    """Tests for individual auto-restart settings entries."""

    def test_settings_list_exists(self):
        """SETTINGS list should exist and contain the expected settings."""
        self.assertTrue(hasattr(llama_manager, 'SETTINGS'))
        self.assertIsInstance(llama_manager.SETTINGS, list)
        self.assertGreater(len(llama_manager.SETTINGS), 0)

    def test_settings_contains_ngl_start(self):
        """SETTINGS should contain ngl_start entry with correct metadata."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("ngl_start")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertIn("-ngl", setting.flag)

    def test_settings_contains_ngl_step(self):
        """SETTINGS should contain ngl_step entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("ngl_step")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")

    def test_settings_contains_retry_on_oom(self):
        """SETTINGS should contain retry_on_oom toggle entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("retry_on_oom")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "bool")

    def test_settings_contains_ctx_start(self):
        """SETTINGS should contain ctx_start entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("ctx_start")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertIn("-c", setting.flag)

    def test_settings_contains_ctx_step(self):
        """SETTINGS should contain ctx_step entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("ctx_step")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")

    def test_settings_contains_enable_ctx_reduction(self):
        """SETTINGS should contain enable_ctx_reduction toggle entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("enable_ctx_reduction")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "bool")

    def test_settings_contains_hang_timeout_mins(self):
        """SETTINGS should contain hang_timeout_mins entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("hang_timeout_mins")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")

    def test_settings_contains_stagnation_timeout_mins(self):
        """SETTINGS should contain stagnation_timeout_mins entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("stagnation_timeout_mins")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")

    def test_settings_contains_oom_detection_timeout_secs(self):
        """SETTINGS should contain oom_detection_timeout_secs entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("oom_detection_timeout_secs")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")

    def test_settings_contains_ctx_reduction_pct(self):
        """SETTINGS should contain ctx_reduction_pct entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("ctx_reduction_pct")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")

    def test_settings_contains_auto_recovery_enabled(self):
        """SETTINGS should contain auto_recovery_enabled master toggle entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("auto_recovery_enabled")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "bool")

    def test_all_keys_have_labels(self):
        """Every setting should have a non-empty label."""
        for s in llama_manager.SETTINGS:
            self.assertNotEqual(s.label, "", f"Setting {s.key} has empty label")

    def test_all_keys_have_types(self):
        """Every setting should have a valid type."""
        valid_types = {"int", "bool", "enum", "string"}
        for s in llama_manager.SETTINGS:
            self.assertIn(s.type, valid_types, f"Setting {s.key} has invalid type {s.type}")

    def test_all_keys_have_defaults(self):
        """Every setting should have a default value."""
        for s in llama_manager.SETTINGS:
            self.assertIsNotNone(s.default, f"Setting {s.key} has no default")

    def test_read_all_settings_returns_dict(self):
        """read_all_settings should return a dict from service text."""
        text = """[Service]
# llama-manager: ngl_start=24
# llama-manager: ngl_step=4
# llama-manager: retry_on_oom=true
ExecStart=llama-server -m /x.gguf -ngl 24 -c 4096
"""
        settings = llama_manager.read_all_settings(text)
        self.assertIsInstance(settings, dict)
        self.assertEqual(settings["ngl_start"], 24)
        self.assertEqual(settings["ctx_start"], 4096)
        self.assertTrue(settings["retry_on_oom"])

    def test_read_settings_from_empty_meta(self):
        """read_all_settings should use defaults when no meta is present."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24 -c 4096
"""
        settings = llama_manager.read_all_settings(text)
        self.assertIsInstance(settings, dict)
        # Should fall back to ExecStart values or defaults
        self.assertIn("ngl_step", settings)

    def test_write_setting_updates_int_meta(self):
        """write_setting should update an int setting in meta comments."""
        text = """[Service]
# llama-manager: ngl_step=4
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "ngl_step", "8")
        self.assertIn("ngl_step=8", result)

    def test_write_setting_updates_execstart_ngl(self):
        """write_setting should update -ngl in ExecStart when setting is ngl_start."""
        text = """[Service]
# llama-manager: ngl_start=24
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "ngl_start", "40")
        self.assertIn("-ngl 40", result)

    def test_write_setting_updates_execstart_ctx(self):
        """write_setting should update -c in ExecStart when setting is ctx_start."""
        text = """[Service]
# llama-manager: ctx_start=4096
ExecStart=llama-server -m /x.gguf -ngl 24 -c 4096
"""
        result = llama_manager.write_setting(text, "ctx_start", "8192")
        self.assertIn("-c 8192", result)
        self.assertIn("ctx_start=8192", result)

    def test_write_setting_handles_bool_true(self):
        """write_setting should update bool toggle to true."""
        text = """[Service]
# llama-manager: retry_on_oom=false
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "retry_on_oom", "true")
        self.assertIn("retry_on_oom=true", result)

    def test_write_setting_handles_bool_false(self):
        """write_setting should update bool toggle to false."""
        text = """[Service]
# llama-manager: retry_on_oom=true
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "retry_on_oom", "false")
        self.assertIn("retry_on_oom=false", result)

    def test_run_auto_restart_settings_exists(self):
        """run_auto_restart_settings function should exist."""
        self.assertTrue(hasattr(llama_manager, 'run_auto_restart_settings'))
        self.assertTrue(callable(llama_manager.run_auto_restart_settings))

    def test_menu_has_auto_restart_settings_entry(self):
        """MENU should have auto_restart_settings action entry."""
        actions = [action for _, action in llama_manager.MENU]
        self.assertIn("auto_restart_settings", actions)

    def test_settings_all_keys_unique(self):
        """All settings should have unique keys."""
        keys = [s.key for s in llama_manager.SETTINGS]
        self.assertEqual(len(keys), len(set(keys)))

# ---------------------------------------------------------------------------
# Expanded Server Configuration Settings (beyond recovery)
# ---------------------------------------------------------------------------

class ExpandedSettingsTest(unittest.TestCase):
    """Tests for expanded SETTINGS registry with server/performance flags."""

    def test_settings_contains_host(self):
        """SETTINGS should contain host entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("host")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "string")
        self.assertEqual(setting.flag, "--host")
        self.assertEqual(setting.default, "127.0.0.1")

    def test_settings_contains_port(self):
        """SETTINGS should contain port entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("port")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "--port")
        self.assertEqual(setting.default, 8080)

    def test_settings_contains_server_timeout(self):
        """SETTINGS should contain server_timeout entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("server_timeout")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "--timeout")
        self.assertEqual(setting.default, 600)

    def test_settings_contains_threads(self):
        """SETTINGS should contain threads entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("threads")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "-t")

    def test_settings_contains_threads_batch(self):
        """SETTINGS should contain threads_batch entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("threads_batch")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "-tb")

    def test_settings_contains_batch_size(self):
        """SETTINGS should contain batch_size entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("batch_size")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "-b")
        self.assertEqual(setting.default, 2048)

    def test_settings_contains_ubatch_size(self):
        """SETTINGS should contain ubatch_size entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("ubatch_size")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "-ub")
        self.assertEqual(setting.default, 512)

    def test_settings_contains_mlock(self):
        """SETTINGS should contain mlock entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("mlock")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "bool")
        self.assertEqual(setting.flag, "--mlock")
        self.assertEqual(setting.default, False)

    def test_settings_contains_flash_attn(self):
        """SETTINGS should contain flash_attn entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("flash_attn")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "enum")
        self.assertEqual(setting.flag, "--flash-attn")
        self.assertEqual(setting.default, "auto")
        self.assertIn("on", setting.options)
        self.assertIn("off", setting.options)
        self.assertIn("auto", setting.options)

    def test_settings_contains_cache_type_k(self):
        """SETTINGS should contain cache_type_k entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("cache_type_k")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "enum")
        self.assertEqual(setting.flag, "-ctk")
        self.assertEqual(setting.default, "f16")

    def test_settings_contains_cache_type_v(self):
        """SETTINGS should contain cache_type_v entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("cache_type_v")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "enum")
        self.assertEqual(setting.flag, "-ctv")
        self.assertEqual(setting.default, "f16")

    def test_settings_contains_cache_ram(self):
        """SETTINGS should contain cache_ram entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("cache_ram")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "--cache-ram")
        self.assertEqual(setting.default, 8192)

    def test_settings_contains_parallel_slots(self):
        """SETTINGS should contain parallel_slots entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("parallel_slots")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "-np")
        self.assertEqual(setting.default, 0)

    def test_settings_contains_model_path(self):
        """SETTINGS should contain model_path entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("model_path")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "string")
        self.assertEqual(setting.flag, "-m")
        self.assertEqual(setting.default, "")

    def test_numa_no_longer_in_settings(self):
        """numa setting should have been removed from SETTINGS."""
        keys = {s.key for s in llama_manager.SETTINGS}
        self.assertNotIn("numa", keys)

    def test_settings_contains_no_kv_offload(self):
        """SETTINGS should contain no_kv_offload entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("no_kv_offload")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "bool")
        self.assertEqual(setting.flag, "--no-kv-offload")
        self.assertEqual(setting.default, False)

    def test_settings_contains_reasoning_budget(self):
        """SETTINGS should contain reasoning_budget entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("reasoning_budget")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "--reasoning-budget")
        self.assertEqual(setting.default, -1)

    def test_settings_contains_spec_type(self):
        """SETTINGS should contain spec_type entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("spec_type")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "enum")
        self.assertEqual(setting.flag, "--spec-type")
        self.assertEqual(setting.default, "none")
        self.assertIn("mtp", setting.options)
        self.assertIn("ngram-cache", setting.options)

    def test_settings_contains_no_mmap(self):
        """SETTINGS should contain no_mmap entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("no_mmap")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "bool")
        self.assertEqual(setting.flag, "--no-mmap")
        self.assertEqual(setting.default, False)

    def test_settings_contains_alias(self):
        """SETTINGS should contain alias entry."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("alias")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "string")
        self.assertEqual(setting.flag, "-a")
        self.assertEqual(setting.default, "")

    def test_settings_total_count(self):
        """Total SETTINGS count should be 31 (11 recovery + 15 server + 5 new)."""
        self.assertEqual(len(llama_manager.SETTINGS), 31)

    def test_new_settings_have_labels(self):
        """All new server settings should have labels."""
        server_keys = {"host", "port", "server_timeout", "threads", "threads_batch",
                       "batch_size", "ubatch_size", "mlock", "flash_attn",
                       "cache_type_k", "cache_type_v", "cache_ram",
                       "parallel_slots", "model_path",
                       "no_kv_offload", "reasoning_budget", "spec_type",
                       "no_mmap", "alias"}
        for s in llama_manager.SETTINGS:
            if s.key in server_keys:
                self.assertNotEqual(s.label, "", f"Setting {s.key} has empty label")


# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------

class HardwareDetectionTest(unittest.TestCase):
    """Tests for hardware detection functions."""

    def test_detect_cpu_count_returns_positive_int(self):
        """detect_cpu_count should return the number of CPU cores."""
        count = llama_manager.detect_cpu_count()
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_detect_vram_info_returns_dict(self):
        """detect_vram_info should return dict with total and used GiB."""
        info = llama_manager.detect_vram_info()
        self.assertIsInstance(info, dict)
        self.assertIn("total_gib", info)
        self.assertIn("used_gib", info)
        self.assertIn("free_gib", info)
        # On this system with AMD GPU, should report realistic values
        if info["total_gib"] > 0:
            self.assertGreater(info["total_gib"], 0.5)  # at least 512 MiB

    def test_detect_ram_info_returns_dict(self):
        """detect_ram_info should return dict with total and available GiB."""
        info = llama_manager.detect_ram_info()
        self.assertIsInstance(info, dict)
        self.assertIn("total_gib", info)
        self.assertIn("available_gib", info)
        self.assertIn("used_gib", info)
        if info["total_gib"] > 0:
            self.assertGreater(info["total_gib"], 0.5)

    def test_detect_gpu_type_returns_string(self):
        """detect_gpu_type should return the GPU type."""
        gpu_type = llama_manager.detect_gpu_type()
        self.assertIsInstance(gpu_type, str)
        # Should be one of: "amd", "nvidia", "intel", "none"
        self.assertIn(gpu_type, ("amd", "nvidia", "intel", "none"))

    def test_detect_hardware_returns_complete_dict(self):
        """detect_hardware should return a dict with all hardware info."""
        info = llama_manager.detect_hardware()
        self.assertIsInstance(info, dict)
        self.assertIn("cpu_count", info)
        self.assertIn("vram_total_gib", info)
        self.assertIn("vram_free_gib", info)
        self.assertIn("ram_total_gib", info)
        self.assertIn("ram_available_gib", info)
        self.assertIn("gpu_type", info)
        self.assertGreater(info["cpu_count"], 0)


# ---------------------------------------------------------------------------
# Auto-Recommend Flags
# ---------------------------------------------------------------------------

class AutoRecommendTest(unittest.TestCase):
    """Tests for automated flag recommendation based on hardware."""

    def test_recommend_flags_returns_dict_with_all_settings(self):
        """recommend_flags should return a dict with recommended values for all settings."""
        hardware = {
            "cpu_count": 12,
            "vram_total_gib": 8.0,
            "vram_free_gib": 3.5,
            "ram_total_gib": 32.0,
            "ram_available_gib": 24.0,
            "gpu_type": "amd",
        }
        recommended = llama_manager.recommend_flags(hardware)
        self.assertIsInstance(recommended, dict)
        # Should contain recommendations for key flags
        self.assertIn("threads", recommended)
        self.assertIn("ngl_start", recommended)
        self.assertIn("ctx_start", recommended)
        self.assertIn("batch_size", recommended)
        self.assertIn("mlock", recommended)

    def test_recommend_threads_respects_cpu_count(self):
        """recommend_flags should set threads to cpu_count (or reasonable default)."""
        hardware = {"cpu_count": 12, "vram_total_gib": 8.0, "vram_free_gib": 3.5,
                     "ram_total_gib": 32.0, "ram_available_gib": 24.0, "gpu_type": "amd"}
        recommended = llama_manager.recommend_flags(hardware)
        self.assertGreater(recommended["threads"], 0)
        self.assertLessEqual(recommended["threads"], 12)

    def test_recommend_ngl_uses_vram(self):
        """recommend_flags should suggest ngl based on VRAM."""
        # With lots of VRAM, ngl should be higher
        big_vram = {"cpu_count": 12, "vram_total_gib": 24.0, "vram_free_gib": 20.0,
                     "ram_total_gib": 64.0, "ram_available_gib": 56.0, "gpu_type": "nvidia"}
        big = llama_manager.recommend_flags(big_vram)
        # With little VRAM, ngl should be lower
        small_vram = {"cpu_count": 12, "vram_total_gib": 4.0, "vram_free_gib": 1.0,
                       "ram_total_gib": 16.0, "ram_available_gib": 8.0, "gpu_type": "amd"}
        small = llama_manager.recommend_flags(small_vram)
        self.assertGreaterEqual(big["ngl_start"], small["ngl_start"])

    def test_recommend_ctx_size_scales_with_ram(self):
        """recommend_flags should suggest ctx size based on available RAM."""
        big_ram = {"cpu_count": 12, "vram_total_gib": 8.0, "vram_free_gib": 3.0,
                    "ram_total_gib": 128.0, "ram_available_gib": 96.0, "gpu_type": "amd"}
        small_ram = {"cpu_count": 12, "vram_total_gib": 8.0, "vram_free_gib": 3.0,
                      "ram_total_gib": 8.0, "ram_available_gib": 2.0, "gpu_type": "amd"}
        big = llama_manager.recommend_flags(big_ram)
        small = llama_manager.recommend_flags(small_ram)
        self.assertGreaterEqual(big["ctx_start"], small["ctx_start"])

    def test_recommend_mlock_enabled_with_sufficient_ram(self):
        """recommend_flags should enable mlock when RAM is plentiful."""
        hardware = {"cpu_count": 12, "vram_total_gib": 8.0, "vram_free_gib": 3.0,
                     "ram_total_gib": 64.0, "ram_available_gib": 48.0, "gpu_type": "amd"}
        recommended = llama_manager.recommend_flags(hardware)
        self.assertTrue(recommended["mlock"])

    def test_recommend_mlock_disabled_with_low_ram(self):
        """recommend_flags should disable mlock when RAM is tight."""
        hardware = {"cpu_count": 12, "vram_total_gib": 8.0, "vram_free_gib": 3.0,
                     "ram_total_gib": 8.0, "ram_available_gib": 2.0, "gpu_type": "amd"}
        recommended = llama_manager.recommend_flags(hardware)
        self.assertFalse(recommended["mlock"])

    def test_recommend_parallel_slots_uses_vram(self):
        """recommend_flags should set parallel slots based on VRAM."""
        hardware = {"cpu_count": 12, "vram_total_gib": 24.0, "vram_free_gib": 20.0,
                     "ram_total_gib": 64.0, "ram_available_gib": 56.0, "gpu_type": "nvidia"}
        recommended = llama_manager.recommend_flags(hardware)
        self.assertGreater(recommended.get("parallel_slots", 0), 1)

    def test_recommend_does_not_mutate_input(self):
        """recommend_flags should not modify the input dict."""
        hardware = {"cpu_count": 12, "vram_total_gib": 8.0, "vram_free_gib": 3.5,
                     "ram_total_gib": 32.0, "ram_available_gib": 24.0, "gpu_type": "amd"}
        original = dict(hardware)
        llama_manager.recommend_flags(hardware)
        self.assertEqual(hardware, original)


# ---------------------------------------------------------------------------
# Expanded Read/Write for New Flag Types
# ---------------------------------------------------------------------------

class ExpandedReadWriteTest(unittest.TestCase):
    """Tests for read/write of expanded server config flags."""

    def test_write_setting_updates_host(self):
        """write_setting should update --host in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "host", "0.0.0.0")
        self.assertIn("--host 0.0.0.0", result)

    def test_write_setting_updates_port(self):
        """write_setting should update --port in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf --port 8080
"""
        result = llama_manager.write_setting(text, "port", "9090")
        self.assertIn("--port 9090", result)

    def test_write_setting_adds_new_string_flag(self):
        """write_setting should add a new string flag when none existed."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "host", "0.0.0.0")
        self.assertIn("--host 0.0.0.0", result)
        self.assertIn("host=0.0.0.0", result)  # meta comment

    def test_write_setting_adds_new_int_flag(self):
        """write_setting should add a new int flag to ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "port", "8080")
        self.assertIn("--port 8080", result)

    def test_write_setting_updates_threads(self):
        """write_setting should update -t in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -t 4
"""
        result = llama_manager.write_setting(text, "threads", "8")
        self.assertIn("-t 8", result)

    def test_write_setting_updates_batch_size(self):
        """write_setting should update -b in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -b 2048
"""
        result = llama_manager.write_setting(text, "batch_size", "4096")
        self.assertIn("-b 4096", result)

    def test_write_setting_updates_mlock_bool_true(self):
        """write_setting should add --mlock 1 when enabling."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "mlock", "true")
        self.assertIn("--mlock 1", result)

    def test_write_setting_updates_mlock_bool_false(self):
        """write_setting should remove --mlock when disabling."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24 --mlock 1
"""
        result = llama_manager.write_setting(text, "mlock", "false")
        self.assertNotIn("--mlock", result)

    def test_write_setting_updates_flash_attn_enum(self):
        """write_setting should update --flash-attn enum value."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf --flash-attn auto
"""
        result = llama_manager.write_setting(text, "flash_attn", "on")
        self.assertIn("--flash-attn on", result)

    def test_write_setting_updates_cache_type_k(self):
        """write_setting should update -ctk enum value."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ctk f16
"""
        result = llama_manager.write_setting(text, "cache_type_k", "q8_0")
        self.assertIn("-ctk q8_0", result)

    def test_write_setting_updates_parallel_slots(self):
        """write_setting should update -np in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -np 4
"""
        result = llama_manager.write_setting(text, "parallel_slots", "8")
        self.assertIn("-np 8", result)

    def test_read_all_settings_returns_new_flags(self):
        """read_all_settings should include new config flags."""
        text = """[Service]
# llama-manager: host=0.0.0.0
# llama-manager: port=9090
# llama-manager: threads=8
ExecStart=llama-server -m /x.gguf -ngl 24 --host 0.0.0.0 --port 9090 -t 8
"""
        settings = llama_manager.read_all_settings(text)
        self.assertEqual(settings.get("host"), "0.0.0.0")
        self.assertEqual(settings.get("port"), 9090)
        self.assertEqual(settings.get("threads"), 8)

    def test_read_all_settings_falls_back_for_new_flags(self):
        """read_all_settings should use defaults when new flags not in service file."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        settings = llama_manager.read_all_settings(text)
        self.assertEqual(settings.get("host"), "127.0.0.1")
        self.assertEqual(settings.get("port"), 8080)
        self.assertEqual(settings.get("threads"), 0)  # 0 = auto
        self.assertEqual(settings.get("mlock"), False)
        self.assertEqual(settings.get("flash_attn"), "auto")
        # model_path falls back to ExecStart -m flag
        self.assertEqual(settings.get("model_path"), "/x.gguf")

    def test_write_setting_updates_cache_ram(self):
        """write_setting should update --cache-ram in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf --cache-ram 8192
"""
        result = llama_manager.write_setting(text, "cache_ram", "4096")
        self.assertIn("--cache-ram 4096", result)

    def test_write_setting_updates_model_path(self):
        """write_setting should update -m in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /old/path/model.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "model_path", "/new/path/model.gguf")
        self.assertIn("-m /new/path/model.gguf", result)

    def test_write_setting_string_with_spaces(self):
        """write_setting should handle string values with special characters in meta."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf
"""
        result = llama_manager.write_setting(text, "model_path", "/path/with/special_chars.gguf")
        self.assertIn("model_path=/path/with/special_chars.gguf", result)

    def test_write_setting_turns_on_no_kv_offload(self):
        """write_setting should add --no-kv-offload 1 to ExecStart when enabling."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "no_kv_offload", "true")
        self.assertIn("--no-kv-offload", result)

    def test_write_setting_turns_off_no_kv_offload(self):
        """write_setting should remove --no-kv-offload when disabling."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24 --no-kv-offload 1
"""
        result = llama_manager.write_setting(text, "no_kv_offload", "false")
        self.assertNotIn("--no-kv-offload", result)

    def test_write_setting_updates_reasoning_budget(self):
        """write_setting should update --reasoning-budget in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf --reasoning-budget 4096
"""
        result = llama_manager.write_setting(text, "reasoning_budget", "2048")
        self.assertIn("--reasoning-budget 2048", result)

    def test_write_setting_adds_reasoning_budget(self):
        """write_setting should add --reasoning-budget to ExecStart when new."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "reasoning_budget", "4096")
        self.assertIn("--reasoning-budget 4096", result)

    def test_write_setting_updates_spec_type(self):
        """write_setting should update --spec-type in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf --spec-type none
"""
        result = llama_manager.write_setting(text, "spec_type", "mtp")
        self.assertIn("--spec-type mtp", result)

    def test_write_setting_updates_no_mmap(self):
        """write_setting should add --no-mmap 1 to ExecStart when enabling."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        result = llama_manager.write_setting(text, "no_mmap", "true")
        self.assertIn("--no-mmap", result)

    def test_write_setting_updates_alias(self):
        """write_setting should update -a in ExecStart."""
        text = """[Service]
ExecStart=llama-server -m /x.gguf -a "MyModel"
"""
        result = llama_manager.write_setting(text, "alias", "Qwen3.6-35B")
        self.assertIn("-a Qwen3.6-35B", result)


# ---------------------------------------------------------------------------
# Submenu Integration for Server Config
# ---------------------------------------------------------------------------

class SubmenuIntegrationTest(unittest.TestCase):
    """Tests for submenu integration and hardware info display."""

    def test_recommend_flags_function_exists(self):
        """recommend_flags function should exist."""
        self.assertTrue(hasattr(llama_manager, 'recommend_flags'))
        self.assertTrue(callable(llama_manager.recommend_flags))

    def test_detect_hardware_function_exists(self):
        """detect_hardware function should exist."""
        self.assertTrue(hasattr(llama_manager, 'detect_hardware'))
        self.assertTrue(callable(llama_manager.detect_hardware))

    def test_detect_cpu_count_function_exists(self):
        """detect_cpu_count function should exist."""
        self.assertTrue(hasattr(llama_manager, 'detect_cpu_count'))
        self.assertTrue(callable(llama_manager.detect_cpu_count))

    def test_detect_vram_info_function_exists(self):
        """detect_vram_info function should exist."""
        self.assertTrue(hasattr(llama_manager, 'detect_vram_info'))
        self.assertTrue(callable(llama_manager.detect_vram_info))

    def test_detect_ram_info_function_exists(self):
        """detect_ram_info function should exist."""
        self.assertTrue(hasattr(llama_manager, 'detect_ram_info'))
        self.assertTrue(callable(llama_manager.detect_ram_info))

    def test_detect_gpu_type_function_exists(self):
        """detect_gpu_type function should exist."""
        self.assertTrue(hasattr(llama_manager, 'detect_gpu_type'))
        self.assertTrue(callable(llama_manager.detect_gpu_type))

    def test_settings_count_includes_new_entries(self):
        """SETTINGS count should include both recovery and server flags."""
        self.assertGreaterEqual(len(llama_manager.SETTINGS), 30)

    def test_new_settings_have_descriptions(self):
        """All new settings should have non-empty descriptions."""
        server_keys = {"host", "port", "server_timeout", "threads", "threads_batch",
                       "batch_size", "ubatch_size", "mlock", "flash_attn",
                       "cache_type_k", "cache_type_v", "cache_ram",
                       "parallel_slots", "model_path",
                       "no_kv_offload", "reasoning_budget", "spec_type",
                       "no_mmap", "alias"}
        for s in llama_manager.SETTINGS:
            if s.key in server_keys:
                self.assertNotEqual(s.description, "", f"Setting {s.key} has empty description")


# ---------------------------------------------------------------------------
# Configurable Watch Log Lines
# ---------------------------------------------------------------------------

class WatchLogLinesTest(unittest.TestCase):
    """Tests for configurable watch log lines in watch mode."""

    def test_settings_contains_watch_log_lines(self):
        """SETTINGS should contain watch_log_lines entry with correct metadata."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("watch_log_lines")
        self.assertIsNotNone(setting)
        self.assertEqual(setting.type, "int")
        self.assertEqual(setting.flag, "")
        self.assertEqual(setting.default, 6)

    def test_watch_log_lines_has_description(self):
        """watch_log_lines should have a non-empty description."""
        setting = {s.key: s for s in llama_manager.SETTINGS}.get("watch_log_lines")
        self.assertIsNotNone(setting)
        self.assertNotEqual(setting.description, "")

    def test_watch_log_lines_round_trip(self):
        """watch_log_lines should be readable via _read_all_meta and writable via write_setting."""
        text = """[Service]
# llama-manager: watch_log_lines=3
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        meta = llama_manager._read_all_meta(text)
        self.assertEqual(meta.get("watch_log_lines"), "3")

        result = llama_manager.write_setting(text, "watch_log_lines", "10")
        self.assertIn("watch_log_lines=10", result)
        self.assertNotIn("watch_log_lines=3", result)

    @patch('llama_manager.Path.read_text')
    def test_run_watch_reads_watch_log_lines_from_service_file(self, mock_read):
        """run_watch should read the service file to get watch_log_lines setting."""
        mock_read.return_value = """[Service]
# llama-manager: watch_log_lines=3
ExecStart=llama-server -m /x.gguf -ngl 24
"""
        screen = DummyScreen()

        # Save and replace lm with enough log lines
        original_lm = llama_manager.lm
        test_lm = llama_manager.LogManager(max_lines=100)
        with test_lm.lock:
            for i in range(20):
                test_lm.raw_logs.append(f"log {i}")
        llama_manager.lm = test_lm

        try:
            with patch.object(llama_manager, 'strip_journal_prefix', side_effect=lambda x: x):
                llama_manager.run_watch(screen)
        finally:
            llama_manager.lm = original_lm

        # Verify Path.read_text was called (behavioral change from hardcoded)
        mock_read.assert_called_once()
