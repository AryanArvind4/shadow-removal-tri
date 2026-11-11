import unittest

import numpy as np

from focused_pipeline import Settings, local_adjustment, process


class FocusTests(unittest.TestCase):
    def test_bounded_finite_fields_and_unchanged_input(self):
        raw = np.random.default_rng(5).uniform(40, 150, (128, 160)).astype(np.float32)
        before = raw.copy()
        result, fields = local_adjustment(raw, 2., 45., 65.)
        np.testing.assert_array_equal(raw, before)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.shape, raw.shape)
        self.assertLessEqual(float(np.abs(fields['added_detail']).max()), 10.)
        self.assertLessEqual(float(fields['midtone_lift'].max()), 8.)
        self.assertGreaterEqual(float(fields['midtone_lift'].min()), 0.)
        self.assertGreaterEqual(float(fields['gain'].min()), 1.)
        self.assertLessEqual(float(fields['gain'].max()), 1.75)

    def test_signed_features_at_several_sizes(self):
        for radius in (1, 4, 12, 24):
            with self.subTest(radius=radius):
                raw = np.full((160, 160), 90., np.float32)
                yy, xx = np.indices(raw.shape)
                dark = (xx-55)**2 + (yy-80)**2 <= radius**2
                bright = (xx-110)**2 + (yy-80)**2 <= radius**2
                raw[dark] -= 15
                raw[bright] += 15
                result, _ = local_adjustment(raw, 2., 50., 80.)
                self.assertLess(float(result[dark].mean()), float(result[~(dark|bright)].mean()))
                self.assertGreater(float(result[bright].mean()), float(result[~(dark|bright)].mean()))

    def test_constant_input_not_fabricated(self):
        raw = np.full((96, 128), 75., np.float32)
        view, fields, _ = process(raw)
        np.testing.assert_array_equal(view, raw)
        np.testing.assert_array_equal(fields['corrected'], raw)

    def test_no_adjustment_when_disabled(self):
        raw = np.full((96, 128), 100., np.float32)
        raw[40:50, 40:50] = 95
        result, _ = local_adjustment(raw, 2., 20., 40., Settings(maximum_gain=1.))
        np.testing.assert_array_equal(raw, result)

    def test_invalid_data_and_settings_rejected(self):
        raw = np.full((96, 128), 75., np.float32)
        bad = raw.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            local_adjustment(bad, 2., 50., 80.)
        with self.assertRaises(ValueError):
            local_adjustment(raw, 0., 50., 80.)
        with self.assertRaises(ValueError):
            local_adjustment(raw, 2., 50., 80., Settings(maximum_gain=3.))


if __name__ == '__main__':
    unittest.main()
