import unittest

import numpy as np

import automatic_v5 as auto


class AutomaticTests(unittest.TestCase):
    def scene(self):
        y, x = np.mgrid[:192, :224]
        image = (60 + 12 * x / 224 + 5 * np.sin(y / 50)).astype(np.float32)
        for py in (30, 85, 140):
            for px in (25, 85, 145):
                image[py:py + 30, px:px + 30] += 35
        image[94:100, 94:100] -= 25
        return image

    def test_component_name_does_not_change_processing(self):
        raw = self.scene()
        expected = auto.process(raw, 'U29')[0]['reference_gray']
        actual = auto.process(raw, 'unseen_component')[0]['reference_gray']
        np.testing.assert_array_equal(expected, actual)

    def test_constant_image_not_turned_into_structure(self):
        raw = np.full((96, 128), 83., np.float32)
        result = auto.process(raw)
        np.testing.assert_array_equal(result[0]['reference_gray'], raw)
        self.assertEqual(len(result[2]), 0)
        self.assertAlmostEqual(float(np.std(result[3])), 0., places=4)

    def test_finite_native_resolution_and_bounds(self):
        raw = self.scene()
        outputs, _, _, numeric, info = auto.process(raw)
        for array in [numeric, *outputs.values()]:
            self.assertEqual(array.shape, raw.shape)
            self.assertTrue(np.isfinite(array).all())
        self.assertGreaterEqual(float(outputs['reference_gray'].min()), 0)
        self.assertLessEqual(float(outputs['reference_gray'].max()), 255)
        self.assertLessEqual(info['max_gain'], 1.60001)
        self.assertLessEqual(info['correction_max_abs'], 28.0001)

    def test_synthetic_dark_feature_remains_dark(self):
        raw = self.scene()
        outputs, _, _, numeric, _ = auto.process(raw)
        for a in (numeric, outputs['reference_gray']):
            depth = float(np.median(a[89:93, 94:100]) - np.median(a[95:99, 95:99]))
            self.assertGreater(depth, 18.)

    def test_denoise_reduces_flat_noise_and_limits_large_changes(self):
        raw = (80 + np.random.default_rng(4).normal(0, 2, (128, 128))).astype(np.float32)
        clean, weight = auto.smooth_noise(raw, auto.noise_sigma(raw), .85)
        self.assertLess(float(np.std(clean)), .8 * float(np.std(raw)))
        self.assertTrue(np.all((weight >= 0) & (weight <= .85)))

    def test_tone_is_increasing(self):
        values = auto.tone(np.linspace(0, 255, 256, dtype=np.float32), 70, 100)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_rejects_invalid_input(self):
        for raw in (np.zeros((3, 3)), np.zeros((60, 60, 3)), np.full((40, 40), np.nan),
                    np.full((40, 40), 4095)):
            with self.assertRaises(ValueError):
                auto.process(raw)

    def test_bands_require_evidence(self):
        raw = self.scene()
        field, diagnostic = auto.band_field(raw, 30, 10, 1)
        np.testing.assert_array_equal(field, np.zeros_like(raw))
        self.assertFalse(diagnostic['applied'])


if __name__ == '__main__':
    unittest.main()
