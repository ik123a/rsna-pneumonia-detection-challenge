"""Catch parse failures that disable all research dashboard controls."""
import shutil
import subprocess
import unittest
from pathlib import Path

class DashboardSyntaxTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is required for JavaScript syntax validation')
    def test_dashboard_javascript_parses(self):
        source = Path(__file__).resolve().parents[1] / 'static/js/app.js'
        result = subprocess.run(['node', '--check', str(source)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
