import unittest

import cv2
import numpy as np

import v5b


class ComponentCompositingTests(unittest.TestCase):
    def test_background_exact_and_interior_retains_dark_feature(self):
        rng = np.random.default_rng(4)
        raw = (90+rng.normal(0, 2, (100, 120))).astype(np.float32)
        support = np.zeros(raw.shape, np.uint8)
        cv2.circle(support, (60, 50), 25, 1, -1)
        raw[support > 0] = 150
        raw[46:55, 56:65] = 70
        enhanced = raw*1.2+12
        enhanced[support == 0] -= 40
        output, alpha, _, _ = v5b.composite(raw, enhanced, support > 0, 50)
        np.testing.assert_array_equal(output[support == 0], raw[support == 0])
        np.testing.assert_array_equal(output[46:55, 56:65], enhanced[46:55, 56:65])
        self.assertLess(float(output[50, 60]), float(output[50, 72]))
        self.assertTrue(np.all((alpha >= 0) & (alpha <= 1)))

    def test_boundary_avoids_added_darkening_but_preserves_source_rim(self):
        raw = np.full((80,80), 90., np.float32)
        support = np.zeros(raw.shape, np.uint8)
        cv2.circle(support, (40,40), 20, 1, -1)
        raw[40,61] = 40
        enhanced = raw-30
        output, alpha, _, halo = v5b.composite(raw, enhanced, support > 0, 40)
        self.assertEqual(float(output[40,60]), float(raw[40,60]))
        self.assertEqual(float(output[40,61]), 40.)
        self.assertEqual(float(output[40,40]), float(enhanced[40,40]))
        self.assertGreater(float(halo.max()), 0)

    def test_no_support_returns_source(self):
        raw = np.full((64,64), 78., np.float32)
        output, alpha, _, _ = v5b.composite(raw, raw+50, np.zeros(raw.shape, bool), 20)
        np.testing.assert_array_equal(output, raw)
        self.assertFalse(alpha.any())

    def test_round_support_includes_internal_cavity_without_filling_pixels(self):
        raw = np.full((160,160), 60., np.float32)
        cv2.circle(raw, (80,80), 25, 140., -1)
        cv2.circle(raw, (80,80), 7, 30., -1)
        mask, info = v5b.component_support(raw, None, [], dict(noise_sigma=.5, feature_diameter_native=50.))
        self.assertTrue(mask[80,80])
        self.assertTrue(mask[80,95])
        self.assertFalse(mask[10,10])
        self.assertGreaterEqual(info['count'], 1)
        self.assertEqual(raw[80,80], 30.)

    def test_grid_support_keeps_center_void_and_tracks_translated_pad(self):
        def select(offset):
            raw = np.full((100,120), 60., np.float32)
            x, y, w, h = 35+offset, 30, 32, 32
            cv2.rectangle(raw, (x,y), (x+w-1,y+h-1), 130., -1)
            cv2.circle(raw, (x+16,y+16), 5, 25., -1)
            labels = np.zeros(raw.shape, np.int32)
            labels[y:y+h,x:x+w] = 1
            return v5b.component_support(raw, labels, [(1,x,y,w,h)],
                dict(noise_sigma=.5,feature_diameter_native=32.))[0]
        first, second = select(0), select(9)
        self.assertTrue(first[46,51])
        np.testing.assert_array_equal(first[:, :-9], second[:, 9:])

    def test_constant_image_stays_constant_end_to_end(self):
        raw = np.full((64,64), 91., np.float32)
        result = v5b.process(raw)
        np.testing.assert_array_equal(result['output'], raw)

    def test_invalid_input_rejected(self):
        for raw in (np.zeros((10,10)), np.full((40,40), np.nan), np.full((40,40), 256)):
            with self.assertRaises(ValueError):
                v5b.process(raw)

    def test_rgb_background_and_protected_annotation_unchanged(self):
        source = np.full((80,80,3), [70,75,72], np.uint8)
        source[40,:] = [255,0,0]
        support = np.zeros((80,80), bool)
        support[20:60,20:60] = True
        support[40] = False
        raw = np.mean(source, axis=2).astype(np.float32)
        output, alpha, target, _ = v5b.composite(raw, raw+50, support, 40)
        result = dict(output=output, alpha=alpha, target=target, support=support)
        final = v5b.render_original_mode(source, result)
        np.testing.assert_array_equal(final[~support], source[~support])
        np.testing.assert_array_equal(final[40], source[40])


if __name__ == '__main__':
    unittest.main()
