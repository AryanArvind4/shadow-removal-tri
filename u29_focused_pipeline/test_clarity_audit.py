import unittest
from unittest.mock import patch

import numpy as np

import u29_clarity as clarity
from audit_clarity import tone_control, variants


class TestAudit(unittest.TestCase):
    def test_tone_is_nondecreasing_and_spatially_independent(self):
        values = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = tone_control(values)
        self.assertTrue(np.all(np.diff(out.ravel().astype(int)) >= 0))
        np.testing.assert_array_equal(tone_control(values.T), out.T)

    def test_conservative_disables_detail_operations(self):
        p = variants()["conservative"]
        for name in ("denoise_h", "deconv_sigma", "deconv_iters",
                     "dome_strength", "void_amount", "unsharp_amount"):
            self.assertEqual(getattr(p, name), 0)

    def test_zero_denoising_preserves_float_source(self):
        src = np.arange(160 * 160, dtype=np.float32).reshape(160, 160) % 120 + .25
        axis = np.array([40., 115.], dtype=np.float32)
        centers = np.array([[(x, y) for x in axis] for y in axis], dtype=np.float32)
        grid = (centers, axis, axis, 75., 75.)
        with patch.object(clarity, "detect_grid", return_value=grid):
            _, _, stages = clarity.process("unused", variants()["conservative"], image=src)
        np.testing.assert_array_equal(stages["work"], src)


if __name__ == "__main__":
    unittest.main()
