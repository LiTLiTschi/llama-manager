import unittest
import sys
import os
import re
from pathlib import Path

# Add the project directory to sys.path
sys.path.append('/home/liu/gists/llama-manager')

import llama_manager as lm

class TestLlamaManager(unittest.TestCase):
    def test_strip_journal_prefix(self):
        line = "May 08 20:34:55 buntu llama-server[134484]: load_tensors: offloaded 40/42 layers"
        expected_start = "08 20:34 | load_tensors:"
        result = lm.strip_journal_prefix(line)
        self.assertTrue(result.startswith(expected_start))

    def test_extract_ngl(self):
        text = "ExecStart=/usr/bin/llama-server -ngl 40 --other-args"
        self.assertEqual(lm._extract_execstart_ngl(text), 40)
        
    def test_extract_ctx(self):
        text = "ExecStart=/usr/bin/llama-server -c 2048 --other-args"
        self.assertEqual(lm._extract_execstart_ctx(text), 2048)

    def test_regex_ngl_replacement(self):
        content = "ExecStart=/usr/bin/llama-server -ngl 40 -c 2048"
        # Test NGL update
        new_content = re.sub(r'(\s-ngl\s+)\d+', r'\g<1>39', content)
        self.assertEqual(new_content, "ExecStart=/usr/bin/llama-server -ngl 39 -c 2048")
        
        # Test Context update
        new_content = re.sub(r'(\s-c\s+)\d+', r'\g<1>1024', new_content)
        self.assertEqual(new_content, "ExecStart=/usr/bin/llama-server -ngl 39 -c 1024")

if __name__ == '__main__':
    unittest.main()
