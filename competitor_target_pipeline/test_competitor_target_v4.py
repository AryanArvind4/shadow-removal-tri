import json
import unittest

import numpy as np

from competitor_target_v4 import clean_background, masked_bands, process, square_grid


class ComponentCleanupTests(unittest.TestCase):
    def test_background_noise_falls_without_changing_solder_pixels(self):
        rng = np.random.default_rng(4)
        raw = (80 + rng.normal(0, 3, (160, 180))).astype(np.float32)
        labels = np.zeros(raw.shape, np.int32)
        labels[50:110, 60:120] = 1
        raw[75:80, 85:90] -= 20
        cleaned, weight = clean_background(raw, labels, 3)
        np.testing.assert_array_equal(cleaned[labels > 0], raw[labels > 0])
        self.assertTrue((weight[labels > 0] == 0).all())
        self.assertLess(cleaned[:40].std(), raw[:40].std())
        self.assertLessEqual(float(np.abs(cleaned-raw).max()), 6.)

    def test_background_only_band_estimation_and_component_gate(self):
        yy, xx = np.mgrid[:160, :240]
        raw = (80 + 5*np.sin(yy/5)).astype(np.float32)
        labels = np.zeros(raw.shape, np.int32)
        labels[50:90, 70:110] = 1
        raw[labels > 0] += 50
        field = masked_bands(raw, labels, 'J4')
        self.assertLess((raw+field)[:, :40].mean(1).std(), raw[:, :40].mean(1).std())
        self.assertLessEqual(float(np.abs(field).max()), 8.)
        for component in ['U29', 'U30']:
            self.assertFalse(masked_bands(raw, labels, component).any())

    def test_grid_requires_image_evidence_and_manifest_serializes(self):
        raw = np.full((600, 700), 80., np.float32)
        empty = np.zeros(raw.shape, np.int32)
        _, regions, diagnostic = square_grid(raw, empty, [], 1.)
        self.assertFalse(regions)
        self.assertFalse(diagnostic['used'])
        for y in .115*600 + .11*600*np.arange(8):
            for x in .115*700 + .097*700*np.arange(9):
                x, y = int(x), int(y)
                raw[y-16:y+17, x-16:x+17] = 110
                raw[y-3:y+4, x-3:x+4] = 65
        labels, regions, diagnostic = square_grid(raw, empty, [], 1.)
        self.assertTrue(diagnostic['used'])
        self.assertEqual(len(regions), 72)
        self.assertGreater(labels[69, 80], 0)
        json.dumps(diagnostic)

    def test_constant_input_is_finite_and_native_resolution(self):
        raw = np.full((100, 120), 80., np.float32)
        outputs, labels, regions, numeric, diagnostic = process(raw, 'U30')
        self.assertFalse(regions)
        self.assertTrue(np.isfinite(numeric).all())
        np.testing.assert_allclose(raw, numeric, atol=1e-4)
        for image in outputs.values():
            self.assertEqual(image.shape, raw.shape)
            self.assertTrue(np.isfinite(image).all())
        json.dumps(diagnostic)


if __name__ == '__main__':
    unittest.main()
