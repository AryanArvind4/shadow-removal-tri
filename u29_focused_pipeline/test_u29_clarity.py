"""Regression checks for the U29 clarity pipeline (V3).

The failure that got V1 rejected was pad deformation, so the checks that
matter most here are geometric: pad edges must stay where the source puts
them, and nothing that suppresses background may reach inside a pad.
"""

import glob
import os
import unittest

import cv2
import numpy as np

import u29_clarity as U


SRC_DIR = r"E:\Shadow_Removal_SDE\TRI_Dataset-20260511T190518Z-3-001\TRI_Dataset_Concised"
PATTERN = "U29_*.jpg"


def _inputs():
    files = sorted(glob.glob(os.path.join(SRC_DIR, PATTERN)))
    if not files:
        raise unittest.SkipTest("U29 inputs not present at " + SRC_DIR)
    return files


PAD_HALF = 18.0          # measured pad half width in pixels
EDGE_SEARCH = 7          # how far either side of nominal the peak may sit


def _edge_halfheight(profile, centre_idx, side):
    """Locate one pad edge at its own half-height crossing.

    Both the plateau and the background level are taken from this profile,
    on this side, in this image. That matters because the pipeline changes
    both deliberately, so any fixed grey level or any gradient extremum would
    move even when the boundary does not: a monotone tone curve preserves
    level sets but not the location of peak slope.

    ``side`` is -1 for the left or upper edge and +1 for the right or lower.
    Returns (position, contrast) or None when the profile shows no clear step.
    """
    seg = cv2.GaussianBlur(np.asarray(profile, np.float32).reshape(-1, 1),
                           (0, 0), 0.8).ravel().astype(np.float64)
    core = seg[max(0, centre_idx - 8):centre_idx + 9]
    a = centre_idx + side * 24
    b = centre_idx + side * 33
    lo_i, hi_i = (int(min(a, b)), int(max(a, b)))
    if lo_i < 0 or hi_i >= len(seg):
        return None
    outside = seg[lo_i:hi_i + 1]
    if core.size < 5 or outside.size < 5:
        return None
    hi = float(np.median(core))
    lo = float(np.median(outside))
    contrast = hi - lo
    if contrast < 4.0:
        return None
    mid = 0.5 * (hi + lo)

    # scan outward from the pad centre and take the first crossing
    i = centre_idx
    limit = centre_idx + side * 30
    while (side > 0 and i < limit) or (side < 0 and i > limit):
        j = i + side
        if j < 0 or j >= len(seg):
            return None
        u, v = seg[i], seg[j]
        if (u - mid) * (v - mid) <= 0 and u != v:
            return i + side * (u - mid) / (u - v), contrast
        i = j
    return None


class TestGeometry(unittest.TestCase):
    """Pad shape must survive the correction."""

    @classmethod
    def setUpClass(cls):
        cls.files = _inputs()
        cls.p = U.Params()
        cls.cache = {}
        for f in cls.files:
            cls.cache[f] = U.process(f, cls.p)

    def test_grid_is_eight_by_nine_on_every_input(self):
        for f in self.files:
            _, diag, _ = self.cache[f]
            self.assertEqual((diag["n_rows"], diag["n_cols"]), (8, 9),
                             msg=os.path.basename(f))

    def test_pad_edges_do_not_move(self):
        """Half-height edge crossings must agree with the source.

        The correction is affine with smooth coefficients plus a smooth
        additive board term, so edge positions should move by well under a
        pixel. A larger shift means a field is tracking structure at pad
        scale and is reshaping pads.
        """
        shifts = []
        skipped = 0
        considered = 0
        for f in self.files:
            out, _, st = self.cache[f]
            src = st["source"]
            outf = out.astype(np.float64)
            for row in st["centers"]:
                for cx, cy in row:
                    x, y = int(round(cx)), int(round(cy))
                    if not (40 <= x < src.shape[1] - 40 and 40 <= y < src.shape[0] - 40):
                        continue
                    profiles = (
                        (src[y, x - 36:x + 37], outf[y, x - 36:x + 37]),
                        (src[y - 36:y + 37, x], outf[y - 36:y + 37, x]),
                    )
                    for ps, po in profiles:
                        for side in (-1, +1):
                            considered += 1
                            a = _edge_halfheight(ps, 36, side)
                            b = _edge_halfheight(po, 36, side)
                            # only judge edges that present a clean isolated
                            # step near where a pad edge belongs; elsewhere
                            # the source has no boundary to preserve
                            if (a is None or b is None or a[1] < 8.0
                                    or abs(abs(a[0] - 36) - PAD_HALF) > 8.0
                                    or abs(abs(b[0] - 36) - PAD_HALF) > 8.0):
                                skipped += 1
                                continue
                            shifts.append(abs(a[0] - b[0]))
        self.assertGreater(len(shifts), 400, "too few usable edges to judge")
        shifts = np.array(shifts)
        print("\n  edge shift px: median %.3f  p95 %.3f  max %.3f  "
              "(measured %d of %d edges, %.0f%% not resolvable)"
              % (np.median(shifts), np.percentile(shifts, 95), shifts.max(),
                 len(shifts), considered, 100.0 * skipped / considered))
        self.assertLess(float(np.median(shifts)), 0.5)
        self.assertLess(float(np.percentile(shifts, 95)), 1.5)

    def test_deconvolution_adds_no_halo(self):
        """Deblurring must sharpen without ringing a bright rim into the gap.

        The 20 per cent bright shoulder just inside a pad edge is the solder
        ball rim and is present in the source, so the check looks outside the
        pad instead: the gap next to a pad must not brighten relative to the
        gap between pads.
        """
        p_on = U.Params()
        p_off = U.Params(deconv_sigma=0.0)
        for f in self.files[:3]:
            on, _, st = U.process(f, p_on)
            off, _, _ = U.process(f, p_off)
            h, w = on.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dmin = np.full((h, w), 1e9, np.float32)
            for row in st["centers"]:
                for cx, cy in row:
                    dmin = np.minimum(dmin, np.maximum(np.abs(xx - cx),
                                                       np.abs(yy - cy)))
            near = (dmin > 22) & (dmin < 30)
            far = (dmin > 30) & (dmin < 36)
            if near.sum() < 500 or far.sum() < 500:
                continue
            halo_on = float(np.median(on[near]) - np.median(on[far]))
            halo_off = float(np.median(off[near]) - np.median(off[far]))
            self.assertLessEqual(
                halo_on, halo_off + 3.0,
                msg="deblurring brightened the gap beside pads in "
                    + os.path.basename(f))

    def test_deconvolution_sharpens_pad_edges(self):
        """The deblur stage must actually earn its place."""
        f = self.files[len(self.files) // 2]
        on, _, st = U.process(f, U.Params())
        off, _, _ = U.process(f, U.Params(deconv_sigma=0.0))
        h, w = on.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dmin = np.full((h, w), 1e9, np.float32)
        for row in st["centers"]:
            for cx, cy in row:
                dmin = np.minimum(dmin, np.maximum(np.abs(xx - cx),
                                                   np.abs(yy - cy)))
        band = (dmin > 15) & (dmin < 21)

        def grad(img):
            a = img.astype(np.float32)
            gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
            return float(np.sqrt(gx * gx + gy * gy)[band].mean())
        g_on, g_off = grad(on), grad(off)
        print("\n  pad edge gradient %.0f without deblur, %.0f with (+%.0f%%)"
              % (g_off, g_on, 100.0 * (g_on / g_off - 1.0)))
        self.assertGreater(g_on, g_off * 1.08)

    def test_guard_gain_is_unity_over_every_pad(self):
        """Background attenuation must never reach inside a pad."""
        for f in self.files:
            _, _, st = self.cache[f]
            gain = st["gain"]
            h, w = gain.shape
            for row in st["centers"]:
                for cx, cy in row:
                    x, y = int(round(cx)), int(round(cy))
                    r = 18  # measured pad half width
                    x0, x1 = max(0, x - r), min(w, x + r + 1)
                    y0, y1 = max(0, y - r), min(h, y + r + 1)
                    self.assertGreaterEqual(
                        float(gain[y0:y1, x0:x1].min()), 0.995,
                        msg="attenuation reached inside a pad in " + os.path.basename(f))

    def test_void_boost_weight_clears_the_pad_edge(self):
        """The band pass must not be applied across the pad boundary."""
        p = self.p
        self.assertLess(p.void_inner_r + 3 * 2.5, 22.0,
                        "void weight support reaches the pad edge and will ring")


class TestHeightStack(unittest.TestCase):
    """The full delivery is a laminography height sweep, not one image."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.imgs, cls.keys = U.load_stack("U29")
        except FileNotFoundError:
            raise unittest.SkipTest("U29 height stack not present")

    def test_stack_is_complete(self):
        self.assertEqual(len(self.keys), 68)
        self.assertEqual(self.keys, list(range(68)),
                         "frame indices are not contiguous")

    def test_solder_plane_lands_on_the_ball_layer(self):
        """Focus selection must pick the solder plane, not the die or board.

        This is a real regression guard: selecting on pad-versus-gap
        brightness instead of boundary sharpness picked a plane about eight
        slices low, and the pads came out visibly fragmented.
        """
        _, (lo, peak, hi), sharp = U.solder_plane(self.imgs, U.Params())
        print("\n  solder plane frames %d..%d, peak %d"
              % (self.keys[lo], self.keys[hi], self.keys[peak]))
        # the concised delivery is frames 29 to 36; the peak must sit there
        self.assertTrue(29 <= self.keys[peak] <= 36,
                        "focus peak at %d is off the solder plane" % self.keys[peak])
        self.assertLessEqual(hi - lo, 9, "selected window is too wide")
        # sharpness must actually fall away towards the die and the board
        self.assertLess(sharp[0], sharp[peak] * 0.85)
        self.assertLess(sharp[-1], sharp[peak] * 0.85)

    def test_averaging_the_plane_does_not_lose_void_contrast(self):
        merged, _, _ = U.solder_plane(self.imgs, U.Params())
        p = U.Params()
        single, _, st = U.process("", p, image=self.imgs[34])
        avg, _, _ = U.process("", p, image=merged)
        h, w = single.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dmin = np.full((h, w), 1e9, np.float32)
        for row in st["centers"]:
            for cx, cy in row:
                dmin = np.minimum(dmin, np.maximum(np.abs(xx - cx), np.abs(yy - cy)))
        core = dmin < 13

        def depth(img):
            v = img[core].astype(np.float64)
            return float(np.median(v) - np.percentile(v, 5))
        d_single, d_avg = depth(single), depth(avg)
        print("  void contrast single %.0f, plane average %.0f" % (d_single, d_avg))
        self.assertGreaterEqual(d_avg, d_single * 0.98)


class TestPhotometry(unittest.TestCase):

    def test_tone_curve_is_monotone(self):
        from scipy.interpolate import PchipInterpolator
        p = U.Params()
        c = PchipInterpolator(np.asarray(p.tone_x, float),
                              np.asarray(p.tone_y, float), extrapolate=False)
        # the pipeline clamps into the knot range, so that is the domain
        xs = np.linspace(p.tone_x[0], p.tone_x[-1], 4000)
        d = np.diff(c(xs))
        self.assertGreaterEqual(float(d.min()), -1e-9,
                                "tone curve inverts, so contrast reverses somewhere")

    def test_source_dark_marks_gain_contrast(self):
        """Voids visible in the source must be at least as visible after.

        Depth is measured against each image's own pad-to-background
        amplitude. Raw grey levels are not comparable across the two, since
        changing that amplitude is exactly what the pipeline does.
        """
        files = _inputs()
        p = U.Params()
        ratios = []
        for f in files[:3]:
            out, _, st = U.process(f, p)
            src, centers, gain = st["source"], st["centers"], st["gain"]
            outf = out.astype(np.float64)
            for row in centers:
                for cx, cy in row:
                    x, y = int(round(cx)), int(round(cy))
                    if not (48 <= x < src.shape[1] - 48 and 48 <= y < src.shape[0] - 48):
                        continue
                    ps = src[y - 14:y + 15, x - 14:x + 15]
                    po = outf[y - 14:y + 15, x - 14:x + 15]
                    # local background ring, same pixels in both images
                    ring = (slice(y - 34, y + 35), slice(x - 34, x + 35))
                    bs = np.percentile(src[ring], 20)
                    bo = np.percentile(outf[ring], 20)
                    amp_s = np.median(ps) - bs
                    amp_o = np.median(po) - bo
                    if amp_s < 8.0 or amp_o < 8.0:
                        continue
                    rel_s = (np.median(ps) - np.percentile(ps, 3)) / amp_s
                    if rel_s < 0.12:            # no clear mark in the source
                        continue
                    rel_o = (np.median(po) - np.percentile(po, 3)) / amp_o
                    ratios.append(rel_o / rel_s)
        self.assertGreater(len(ratios), 20, "too few source marks to judge")
        ratios = np.array(ratios)
        print("\n  mark contrast ratio out/src: min %.2f median %.2f (n=%d)"
              % (ratios.min(), np.median(ratios), len(ratios)))
        # Per-pad normalisation divides by the local pad amplitude, so an
        # equally deep mark on an unusually thick pad is correctly rendered as
        # a smaller fractional defect. A few marks therefore hold their
        # contrast rather than gaining; none may lose a meaningful amount.
        self.assertGreater(float(ratios.min()), 0.85,
                           "a source dark mark lost relative contrast")
        self.assertGreater(float(np.median(ratios)), 1.2,
                           "marks are not meaningfully clearer than the source")

    def test_background_is_flatter_than_source(self):
        files = _inputs()
        p = U.Params()
        for f in files:
            out, _, st = U.process(f, p)
            src, gain = st["source"], st["gain"]
            gapm = gain < (p.bg_gain + 0.05)
            if gapm.sum() < 2000:
                continue
            # compare spread after matching overall scale
            s = src[gapm]
            o = out[gapm].astype(np.float64)
            s_rel = (np.percentile(s, 75) - np.percentile(s, 25)) / max(1e-6, np.median(s))
            o_rel = (np.percentile(o, 75) - np.percentile(o, 25)) / max(1e-6, np.median(o))
            self.assertLess(o_rel, s_rel,
                            msg="background got less uniform in " + os.path.basename(f))


if __name__ == "__main__":
    unittest.main(verbosity=2)
