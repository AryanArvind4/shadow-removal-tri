import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import targeted_fixes as fixes


class TargetedFixTests(unittest.TestCase):
    def test_excluded_artifacts_revert_to_source_and_accepted_core_stays(self):
        source = np.full((100,100),70,np.uint8)
        previous = source.copy()
        previous[25:65,25:65] = 160
        previous[85:95,20:90] = 210
        mask = previous != source
        profile = dict(size=[100,100],mode='restrict_existing',feather=3.,boxes=[[20,20,70,70]])
        final,_,support,_,_,_ = fixes.correct(source,previous,previous,mask,profile)
        np.testing.assert_array_equal(final[80:],source[80:])
        np.testing.assert_array_equal(final[30:60,30:60],previous[30:60,30:60])
        np.testing.assert_array_equal(final[~support],source[~support])

    def test_component_mask_encloses_dark_hole_without_filling_image(self):
        source = np.full((90,90),45,np.uint8)
        source[20:70,20:70] = 160
        cv2.circle(source,(45,45),9,30,-1)
        baseline = source.copy()
        baseline[20:70,20:70] = 220
        cv2.circle(baseline,(45,45),9,20,-1)
        profile = dict(size=[90,90],mode='complete_screenshot_components',feather=2.,boxes=[[15,15,75,75]])
        final,_,support,_,_,_ = fixes.correct(source,source,baseline,np.zeros(source.shape,bool),profile)
        self.assertTrue(support[45,45])
        self.assertEqual(final[45,45],20)
        self.assertEqual(final[30,30],220)
        np.testing.assert_array_equal(final[~support],source[~support])

    def test_retained_rgb_regions_and_annotations_are_exact(self):
        source = np.full((90,90,3),50,np.uint8)
        source[20:60,20:60] = 150
        source[35,:] = [255,0,0]
        previous_gray = np.asarray(fixes.Image.fromarray(source).convert('L')).copy()
        previous_color = source.copy()
        previous_color[70:80,70:80] = [180,170,160]
        previous_gray[70:80,70:80] = 172
        old_mask = np.zeros((90,90),bool)
        old_mask[70:80,70:80] = True
        profile = dict(size=[90,90],mode='complete_screenshot_components',feather=2.,
                       boxes=[[15,15,65,65]],retain_previous_outside_boxes=True)
        final,gray,_,_,_,_ = fixes.correct(source,previous_gray,np.minimum(previous_gray.astype(float)+20,255),old_mask,profile,previous_color)
        np.testing.assert_array_equal(final[70:80,70:80],previous_color[70:80,70:80])
        np.testing.assert_array_equal(gray[70:80,70:80],previous_gray[70:80,70:80])
        np.testing.assert_array_equal(final[35],source[35])

    def test_wrong_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.png'
            fixes.Image.fromarray(np.zeros((40,40),np.uint8)).save(path)
            with self.assertRaises(ValueError):
                fixes.validate_source(path,dict(sha256=hashlib.sha256(b'wrong source').hexdigest()))

    def test_wrong_dimensions_are_rejected(self):
        raw = np.zeros((90,90),np.uint8)
        with self.assertRaises(ValueError):
            fixes.correct(raw,raw,raw,raw > 0,dict(size=[80,80]))


if __name__ == '__main__':
    unittest.main()
