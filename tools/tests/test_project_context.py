"""Isolated temporary-repository tests; never touch research inputs or outputs."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'project_context.py'
spec = importlib.util.spec_from_file_location('project_context', SOURCE)
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_git('init', '-q')
        self.run_git('config', 'user.name', 'Context Test')
        self.run_git('config', 'user.email', 'context@example.invalid')
        self.run_git('config', 'core.autocrlf', 'false')
        self.run_git('config', 'commit.gpgsign', 'false')
        self.run_git('config', 'core.hooksPath', 'no-hooks')
        self.write(tracker.HANDOFF, '# Project Handoff\n\nInitial state.\n')
        self.write('module.py', 'value = 1\n')
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'initial')

    def run_git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True)

    def write(self, path, text):
        (self.root / path).write_bytes(text.encode())

    def record(self):
        entry = tracker.record(self.root, 'Explain change', 'Tests passed', 'Await feedback', ['results/local'])
        self.run_git('add', tracker.HANDOFF, tracker.HISTORY)
        return entry

    def test_record_and_check_hashes_timestamp_and_artifacts(self):
        self.write('module.py', 'value = 2\n')
        self.run_git('add', 'module.py')
        entry = self.record()
        self.assertEqual(tracker.check(self.root), entry)
        self.assertTrue(entry['timestamp'].endswith('+08:00'))
        self.assertEqual(entry['files'], {'module.py': tracker.sha(b'value = 2\n')})
        self.assertEqual(entry['artifacts'], ['results/local'])

    def test_missing_context_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Stage both'):
            tracker.check(self.root)

    def test_stale_staged_code_rejected(self):
        self.record()
        self.write('module.py', 'value = 3\n')
        self.run_git('add', 'module.py')
        with self.assertRaisesRegex(ValueError, 'Staged files changed'):
            tracker.check(self.root)

    def test_unstaged_code_not_misrepresented_as_committed(self):
        self.record()
        self.write('module.py', 'value = 9\n')
        self.assertEqual(tracker.check(self.root)['files'], {})

    def test_stale_handoff_rejected(self):
        self.record()
        self.write(tracker.HANDOFF, '# Changed handoff\n')
        self.run_git('add', tracker.HANDOFF)
        with self.assertRaisesRegex(ValueError, 'handoff does not match'):
            tracker.check(self.root)

    def test_new_and_deleted_files(self):
        self.write('new name.py', 'new = True\n')
        self.run_git('rm', 'module.py')
        self.run_git('add', 'new name.py')
        entry = self.record()
        self.assertIsNone(entry['files']['module.py'])
        self.assertEqual(len(entry['files']), 2)
        tracker.check(self.root)

    def test_history_append_and_no_rewrite(self):
        self.record()
        self.run_git('commit', '-qm', 'first record')
        old = (self.root / tracker.HISTORY).read_bytes()
        self.record()
        self.assertTrue((self.root / tracker.HISTORY).read_bytes().startswith(old))
        tracker.check(self.root)
        self.write(tracker.HISTORY, '{}\n')
        self.run_git('add', tracker.HISTORY)
        with self.assertRaisesRegex(ValueError, 'without rewriting'):
            tracker.check(self.root)

    def test_context_only_followup_allowed(self):
        self.record()
        self.run_git('commit', '-qm', 'first record')
        self.write(tracker.HANDOFF, '# Project Handoff\n\nNew feedback.\n')
        self.record()
        tracker.check(self.root)

    def test_record_not_automatically_staged(self):
        tracker.record(self.root, 'Summary', 'Not run', 'Review')
        with self.assertRaisesRegex(ValueError, 'Stage both'):
            tracker.check(self.root)

    def test_hook_blocks_stale_commit_and_allows_current(self):
        hooks = self.root / '.githooks'
        hooks.mkdir()
        source_hook = SOURCE.parents[1] / '.githooks/pre-commit'
        (hooks / 'pre-commit').write_bytes(source_hook.read_bytes())
        (hooks / 'pre-commit').chmod(0o755)
        (self.root / 'tools').mkdir()
        (self.root / 'tools/project_context.py').write_bytes(SOURCE.read_bytes())
        self.run_git('config', 'core.hooksPath', '.githooks')
        self.run_git('add', '.')
        env = dict(os.environ)
        # Use the interpreter running these tests for the hook's PATH fallback.
        import sys
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        args = ['git', 'commit', '-qm', 'guarded']
        denied = subprocess.run(args, cwd=self.root, capture_output=True, env=env)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn(b'Project context check failed', denied.stderr)
        self.record()
        allowed = subprocess.run(args, cwd=self.root, capture_output=True, env=env)
        self.assertEqual(allowed.returncode, 0, allowed.stderr.decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
