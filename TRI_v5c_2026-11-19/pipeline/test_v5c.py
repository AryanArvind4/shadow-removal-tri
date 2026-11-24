import unittest
from dataclasses import replace

import cv2
import numpy as np

import v5c


def fixture(offset=0):
    raw = np.full((110, 130), 60., np.float32)
    x, y, w, h = 32 + offset, 32, 40, 40
    raw[y:y+h, x:x+w] = 140
    cv2.circle(raw, (x+34, y+10), 4, 40., -1)
    regions = [(1, x, y, w, h)]
    baseline = ({'reference_gray': np.minimum(raw+75, 245)},
                np.zeros(raw.shape, np.int32), regions, raw,
                {'baseline_diagnostics': {'noise_sigma': 1., 'feature_diameter_native': 40.},
                 'layout': 'synthetic'}, raw)
    return raw, baseline


class SourceGuidedTests(unittest.TestCase):
    def test_near_edge_void_keeps_source_rim(self):
        raw, baseline = fixture()
        result = v5c.process(raw, baseline=baseline)
        self.assertEqual(result['output'][42, 66], 40.)
        self.assertGreater(result['output'][42, 71], 140.)
        self.assertGreater(result['output'][42, 71] - result['output'][42, 66], 100.)
        self.assertTrue(result['support'][42, 71])

    def test_true_open_notch_is_not_closed(self):
        raw, baseline = fixture()
        raw[40:45, 66:75] = 40
        result = v5c.process(raw, baseline=baseline)
        np.testing.assert_array_equal(result['output'][40:45, 66:75], raw[40:45, 66:75])

    def test_diagonal_open_channel_stays_dark(self):
        raw, baseline = fixture()
        for i in range(9):
            raw[42+i, 66+i] = 40
        result = v5c.process(raw, baseline=baseline)
        for i in range(9):
            self.assertEqual(result['output'][42+i, 66+i], 40)

    def test_source_and_unselected_pixels_are_exact(self):
        raw, baseline = fixture()
        saved = raw.copy()
        result = v5c.process(raw, baseline=baseline)
        np.testing.assert_array_equal(raw, saved)
        np.testing.assert_array_equal(result['output'][~result['support']], raw[~result['support']])
        self.assertTrue(result['details']['background_exact'])

    def test_no_added_darkening_or_overflow(self):
        raw, baseline = fixture()
        result = v5c.process(raw, baseline=baseline)
        self.assertTrue(np.isfinite(result['output']).all())
        self.assertTrue(np.all(result['output'] >= raw))
        self.assertLessEqual(result['output'].max(), 255)
        self.assertTrue(np.all((result['alpha'] >= 0) & (result['alpha'] <= 1)))

    def test_translation(self):
        raw, baseline = fixture()
        first = v5c.process(raw, baseline=baseline)['output']
        raw, baseline = fixture(8)
        second = v5c.process(raw, baseline=baseline)['output']
        np.testing.assert_allclose(first[25:80, 25:80], second[25:80, 33:88], atol=1e-4)

    def test_soft_near_edge_rim_retains_contrast(self):
        raw, baseline = fixture()
        raw = cv2.GaussianBlur(raw, (0,0), .7)
        result = v5c.process(raw, baseline=baseline)
        rim, cavity = (42,71), (42,66)
        self.assertGreater(raw[rim], raw[cavity])
        self.assertGreaterEqual(result['output'][rim]-result['output'][cavity], raw[rim]-raw[cavity])

    def test_sloped_background_remains_unchanged(self):
        raw, baseline = fixture()
        raw += np.linspace(0, 20, raw.shape[1], dtype=np.float32)[None, :]
        result = v5c.process(raw, baseline=baseline)
        np.testing.assert_array_equal(result['output'][:20], raw[:20])
        self.assertGreater(result['output'][60,50], raw[60,50])

    def test_lower_brightness_anchor_does_not_darken(self):
        raw, baseline = fixture()
        baseline[0]['reference_gray'] = raw-30
        result = v5c.process(raw, baseline=baseline)
        np.testing.assert_array_equal(result['output'], raw)
        self.assertFalse(result['support'].any())

    def test_all_protected_is_identity(self):
        raw, baseline = fixture()
        result = v5c.process(raw, baseline=baseline, protected=np.ones(raw.shape,bool))
        np.testing.assert_array_equal(result['output'], raw)
        self.assertFalse(result['support'].any())

    def test_protected_rgb_pixels_remain_exact(self):
        raw, baseline = fixture()
        protected = np.zeros(raw.shape, bool)
        protected[42, :] = True
        result = v5c.process(raw, baseline=baseline, protected=protected)
        source = np.repeat(raw.astype(np.uint8)[..., None], 3, axis=2)
        source[42, :] = [255, 0, 0]
        final = v5c.render_original_mode(source, result)
        np.testing.assert_array_equal(final[~result['support']], source[~result['support']])
        np.testing.assert_array_equal(final[42], source[42])

    def test_constant_end_to_end(self):
        raw = np.full((64, 64), 91., np.float32)
        result = v5c.process(raw)
        np.testing.assert_array_equal(result['output'], raw)
        self.assertFalse(result['support'].any())

    def test_weak_background_rejected(self):
        raw = np.full((100, 100), 60., np.float32)
        raw[30:65, 30:65] = 61
        region, record = v5c.source_region(raw, (30, 30, 35, 35), 1.)
        self.assertIsNone(region)
        self.assertFalse(record['accepted'])

    def test_generic_circle_kept_diffuse_shape_rejected(self):
        raw = np.full((160, 160), 60., np.float32)
        support = np.zeros(raw.shape, np.uint8)
        cv2.circle(support, (45, 50), 20, 1, -1)
        support[90:130, 100:108] = 1
        support[122:130, 100:140] = 1
        proposals = list(v5c.generic_proposals(raw, support))
        self.assertIsNone(proposals[0][2])
        self.assertIsNotNone(proposals[1][2])

    def test_incomplete_background_ring_rejected(self):
        raw = np.full((100, 100), 60., np.float32)
        support = np.zeros(raw.shape, np.uint8)
        support[2:32, 35:65] = 1
        self.assertEqual(list(v5c.generic_proposals(raw, support))[0][2], 'incomplete background context')

    def test_scalar_mapping_order_and_headroom(self):
        source = np.linspace(0, 255, 256).reshape(16, 16).astype(np.float32)
        target, _ = v5c.monotone_lift(source, np.full_like(source, 60), 140, 220, 80)
        self.assertTrue(np.all(np.diff(target.ravel()) >= 0))
        np.testing.assert_array_equal(target[source <= 60], source[source <= 60])
        np.testing.assert_array_equal(target[source >= 245], source[source >= 245])

    def test_invalid_inputs_settings_and_mask(self):
        for raw in (np.zeros((20,20)), np.full((40,40), np.nan), np.full((40,40), 256), np.zeros((40,40,3))):
            with self.assertRaises(ValueError):
                v5c.process(raw)
        raw, baseline = fixture()
        for settings in (replace(v5c.Settings(), weak_fraction=.9), replace(v5c.Settings(), maximum_slope=0)):
            with self.assertRaises(ValueError):
                v5c.process(raw, settings=settings, baseline=baseline)
        with self.assertRaises(ValueError):
            v5c.process(raw, baseline=baseline, protected=np.zeros((2,2)))


if __name__ == '__main__':
    unittest.main()
