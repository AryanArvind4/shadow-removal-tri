"""Checks for detail loss, incorrect ROI metrics, and edge-case inputs."""
import unittest

import numpy as np

from competitor_target_v3 import pad_correction, process, region_metrics, row_correction, tone
from evaluate_reviewed_regions import measure, polygon_mask


class RestorationTests(unittest.TestCase):
    def test_reviewed_polygons_validate_bounds_and_overlap(self):
        with self.assertRaises(ValueError):
            polygon_mask([[0, 0], [20, 0], [0, 20]], (20, 20))
        mask = polygon_mask([[2, 2], [8, 2], [8, 8], [2, 8]], (20, 20))
        with self.assertRaises(ValueError):
            measure(np.zeros((20, 20)), mask, mask)

    def test_reviewed_metrics_keep_contrast_polarity(self):
        healthy = np.zeros((20, 20), bool)
        defect = healthy.copy()
        healthy[1:5, 1:5] = True
        defect[10:14, 10:14] = True
        image = np.full((20, 20), 100., np.float32)
        image[defect] = 60
        measured = measure(image, healthy, defect)
        self.assertEqual(measured['signed_healthy_minus_defect'], 40)
        self.assertIsNone(measured['cnr'])

    def test_native_shape_and_finite_constant(self):
        raw = np.full((91, 113), 84., np.float32)
        output, labels, regions, corrected = process(raw, 'U29')
        self.assertFalse(regions)
        for array in output.values():
            self.assertEqual(array.shape, raw.shape)
            self.assertTrue(np.isfinite(array).all())
        np.testing.assert_allclose(corrected, raw, atol=1e-5)

    def test_rows_disabled_for_u_components(self):
        raw = np.arange(120, dtype=np.float32).reshape(12, 10)
        for component in ('U29', 'U30'):
            out, field = row_correction(raw, component)
            np.testing.assert_array_equal(out, raw)
            self.assertFalse(field.any())

    def test_region_correction_preserves_small_void_depth(self):
        image = np.full((60, 60), 100., np.float32)
        labels = np.zeros((60, 60), np.int32)
        labels[10:50, 10:50] = 1
        image[28:32, 28:32] -= 30
        field = pad_correction(image, labels, [(1, 10, 10, 40, 40)])
        restored = image + field
        self.assertAlmostEqual(float(restored[25, 30] - restored[30, 30]), 30., delta=.25)
        self.assertTrue(np.isfinite(field).all())
        self.assertLessEqual(float(np.max(np.abs(field))), 18.01)
        self.assertFalse(field[labels == 0].any())

    def test_separate_pad_statistics(self):
        image = np.zeros((20, 40), np.float32)
        labels = np.zeros_like(image, dtype=np.int32)
        labels[2:18, 2:18] = 1
        labels[2:18, 22:38] = 2
        image[labels == 1] = 50
        image[labels == 2] = 150
        outputs = {'reference_gray': image, 'detail_gray': image}
        records = region_metrics(image, outputs, labels, [(1, 2, 2, 16, 16), (2, 22, 2, 16, 16)])
        self.assertEqual(len(records), 6)
        self.assertTrue(all(r['interior_std'] == 0 for r in records))
        self.assertGreater(np.std(image[labels > 0]), 0)

    def test_tone_preserves_order_and_headroom(self):
        values = np.linspace(-50, 300, 1000, dtype=np.float32)
        out = tone(values, 30, 180)
        self.assertTrue((np.diff(out) > 0).all())
        self.assertGreater(out.min(), 0)
        self.assertLess(out.max(), 255)

    def test_unknown_structure_is_not_replaced_by_mask(self):
        yy, xx = np.mgrid[:96, :128]
        raw = (70 + 10 * np.sin(xx / 6) + 2 * np.cos(yy / 3)).astype(np.float32)
        outputs, _, _, corrected = process(raw, 'U30')
        self.assertEqual(outputs['reference_gray'].shape, raw.shape)
        self.assertLess(float(np.max(np.abs(corrected - raw))), 34)


if __name__ == '__main__':
    unittest.main()
