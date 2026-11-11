import unittest

import cv2
import numpy as np

import u29_focused as u


class ShapeCorrectionTests(unittest.TestCase):
    def fixture(self):
        yy,xx = np.mgrid[:310,:370]
        raw = (55.+xx*.025+yy*.015).astype(np.float32)
        labels = np.zeros(raw.shape,np.int32)
        regions = []
        xs,ys = list(range(35,336,60)),list(range(35,276,60))
        for y in ys:
            for x in xs:
                i = len(regions)+1
                level = 95.+.055*y
                cv2.rectangle(raw,(x-16,y-12),(x+15,y+11),level,-1)
                cv2.rectangle(raw,(x-12,y-16),(x+11,y+15),level,-1)
                for dx,dy in [(-12,-12),(11,-12),(-12,11),(11,11)]:
                    cv2.circle(raw,(x+dx,y+dy),4,level,-1)
                labels[y-16:y+16,x-16:x+16] = i
                regions.append((i,x-16,y-16,32,32))
        raw = u.base.blur(raw,.7)
        return raw,labels,regions,dict(used=True,diameter=32.,rows=ys,columns=xs)

    def test_no_filtering_or_dot_suppression_at_pad_edges(self):
        raw,labels,regions,layout = self.fixture()
        raw[93:98,79:86] -= 25.  # A genuine notch reaching the left pad edge.
        for y in (65,125,185,245):
            for x in (65,125,185,245,305):
                cv2.circle(raw,(x,y),2,95.,-1)
        images,arrays,_ = u.enhance(raw,labels,regions,layout,1.)
        guard = arrays['edge_guard']
        np.testing.assert_array_equal(arrays['background_delta'][guard],0.)
        np.testing.assert_array_equal(images['reference_gray'][guard],images['detail_gray'][guard])
        self.assertTrue(np.any(np.abs(arrays['background_delta'][~guard])>.1))

    def test_small_and_large_marks_recover_from_photometric_transform(self):
        raw,labels,regions,layout = self.fixture()
        cv2.circle(raw,(95,95),3,40.,-1)
        cv2.circle(raw,(155,155),12,45.,-1)
        raw[215,215],raw[95,215] = 35.,150.
        images,arrays,_ = u.enhance(raw,labels,regions,layout,1.)
        out = images['reference_gray'].astype(np.float64)
        linear = out.copy()
        low,high = out<40.,out>210.
        linear[low] = 40.-32.*(32./(out[low]-8.)-1.)
        linear[high] = 210.+37.*(37./(247.-out[high])-1.)
        settings = u.Settings()
        reconstructed = (linear-settings.display_background)/(settings.display_gain*arrays['gain_field'])+arrays['background_field']
        guard = arrays['edge_guard']
        self.assertLess(float(np.abs(reconstructed[guard]-raw[guard]).max()),.0002)
        self.assertLess(float(out[95,95]),float(out[85,95])-30.)
        self.assertLess(float(out[155,155]),float(out[140,155])-30.)

    def test_rounded_edges_stay_in_place(self):
        raw,labels,regions,layout = self.fixture()
        images,_,_ = u.enhance(raw,labels,regions,layout,1.)
        checks = u.edge_profile_checks(raw,{'v2':images['reference_gray']},regions)
        self.assertGreater(len(checks),500)
        self.assertTrue(all(r['displacement'] is not None for r in checks))
        self.assertLess(float(np.percentile([r['displacement'] for r in checks],95)),.35)

    def test_global_curve_is_strictly_monotone_and_linear_in_working_range(self):
        values = np.linspace(-300,500,2000,dtype=np.float32)
        output = u.display_curve(values)
        self.assertTrue(np.all(np.diff(output)>0))
        self.assertGreater(float(output.min()),8.)
        self.assertLess(float(output.max()),247.)
        middle = np.arange(40,211,dtype=np.float32)
        np.testing.assert_array_equal(u.display_curve(middle),middle)

    def test_field_has_no_jump_at_rectangle_boundary(self):
        raw,labels,regions,layout = self.fixture()
        _,arrays,_ = u.enhance(raw,labels,regions,layout,1.)
        self.assertLess(max(float(np.abs(a).max()) for a in np.gradient(arrays['gain_field'])),.02)
        self.assertLess(max(float(np.abs(a).max()) for a in np.gradient(arrays['background_field'])),.1)

    def test_no_supported_grid_rejects_output(self):
        raw,labels,regions,layout = self.fixture()
        layout['used'] = False
        with self.assertRaises(ValueError):
            u.enhance(raw,labels,regions,layout,1.)

    def test_invalid_and_blank_sources_do_not_make_pads(self):
        for raw in (np.zeros((30,30),np.float32),np.full((80,80),np.nan,np.float32),np.full((96,96),60.,np.float32)):
            with self.assertRaises(ValueError):
                u.process(raw)


if __name__=='__main__':
    unittest.main()
