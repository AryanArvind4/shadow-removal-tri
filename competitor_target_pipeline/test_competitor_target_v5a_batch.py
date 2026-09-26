"""Packaging and BMP adapter checks without private TRI images."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


class BatchPackagingTests(unittest.TestCase):
    root = Path(__file__).resolve().parent

    def test_selected_source_hashes(self):
        manifest = json.loads((self.root / 'v5a_source_manifest.json').read_text())
        for name, expected in manifest['source_hashes'].items():
            with self.subTest(name=name):
                actual = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_bmp_batch_without_historical_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / 'pipeline'
            pipeline.mkdir()
            # Only published runtime files, deliberately no historical results.
            for name in ('run_v5a_bmp_batch.py', 'v5a_source_manifest.json',
                         'competitor_target_v5a.py', 'automatic_v5.py',
                         'competitor_target_v3.py'):
                (pipeline / name).write_bytes((self.root / name).read_bytes())
            source = root / 'input'
            source.mkdir()
            raw = np.full((96, 128), 60, dtype=np.uint8)
            raw[30:60, 40:80] = 100
            image_path = source / 'sample.bmp'
            Image.fromarray(raw).save(image_path)
            original = image_path.read_bytes()
            output = root / 'output'
            command = [sys.executable, str(pipeline / 'run_v5a_bmp_batch.py'),
                       '--input', str(source), '--output', str(output)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            report = json.loads((output / 'source_preparation.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(report['originals_unchanged'])
            self.assertTrue(report['native_output_dimensions_verified'])
            self.assertEqual(image_path.read_bytes(), original)
            with Image.open(output / 'prepared_inputs/sample.png') as prepared:
                np.testing.assert_array_equal(np.asarray(prepared), raw)
            with Image.open(output / 'reference_gray/sample.png') as result:
                self.assertEqual(result.size, (128, 96))
                self.assertEqual(result.mode, 'L')
            again = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('Refusing to overwrite', again.stderr)


if __name__ == '__main__':
    unittest.main()
