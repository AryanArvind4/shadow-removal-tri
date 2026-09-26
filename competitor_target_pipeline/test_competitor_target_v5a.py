import unittest

import cv2
import numpy as np

import competitor_target_v5a as v5a


class SelectivePadTests(unittest.TestCase):
    def fixture(self):
        raw = np.full((360,400), 60., np.float32)
        xs = list(range(40,361,60))
        ys = list(range(40,301,60))
        labels = np.zeros(raw.shape,np.int32)
        regions = []
        for y in ys:
            for x in xs:
                identifier = len(regions)+1
                raw[y-12:y+12,x-12:x+12] = 110 if y < 100 else 90
                labels[y-12:y+12,x-12:x+12] = identifier
                regions.append((identifier,x-12,y-12,24,24))
        return raw, labels, regions, dict(used=True,diameter=24.,columns=xs,rows=ys)

    def test_detects_repeated_grid_without_fixed_count(self):
        raw,_,regions,_ = self.fixture()
        labels,found,info = v5a.rectangular_layout(raw,24,1)
        self.assertTrue(info['used'])
        self.assertEqual(len(found),len(regions))
        self.assertGreater(int(labels.max()),0)

    def test_no_grid_on_blank_or_line_only_image(self):
        raw = np.full((360,400),60.,np.float32)
        for image in (raw, raw + (np.indices(raw.shape)[0] % 60 < 12) * 30):
            labels,regions,info = v5a.rectangular_layout(image.astype(np.float32),24,1)
            self.assertFalse(info['used'])
            self.assertEqual(len(regions),0)
            self.assertFalse(labels.any())

    def test_small_dark_feature_is_not_filled(self):
        raw,labels,regions,_ = self.fixture()
        raw[157:163,157:163] -= 25
        field,records = v5a.pad_field(raw,labels,regions)
        result = raw+field
        self.assertGreater(float(np.mean(result[151:155,157:163])-np.mean(result[158:162,158:162])),23)
        self.assertLessEqual(float(np.abs(field).max()),22.0001)
        self.assertTrue(all(-6 <= r['shift'] <= 18 for r in records))

    def test_repeated_dots_only_and_pad_guard(self):
        raw,labels,_,layout = self.fixture()
        for y in (70,130,190,250):
            for x in (70,130,190,250,310):
                cv2.circle(raw,(x,y),2,95.,-1)
        cv2.circle(raw,(89,82),2,100.,-1)  # Isolated phase, not a repeated dot.
        cv2.circle(raw,(46,46),2,145.,-1)  # Inside a pad.
        mask,spots = v5a.repeated_dots(raw,labels,layout,1)
        self.assertGreater(len(spots),10)
        self.assertGreater(float(mask[70,70]),.3)
        self.assertEqual(float(mask[82,89]),0)
        self.assertEqual(float(mask[labels>0].max()),0)

    def test_filename_independence_and_fallback(self):
        raw = np.full((96,128),75.,np.float32)
        a = v5a.process(raw,'U29')
        b = v5a.process(raw,'unknown')
        np.testing.assert_array_equal(a[0]['reference_gray'],b[0]['reference_gray'])
        np.testing.assert_array_equal(a[0]['reference_gray'],raw)
        self.assertFalse(a[4]['layout']['used'])
        self.assertTrue(np.isfinite(a[3]).all())


if __name__ == '__main__':
    unittest.main()
