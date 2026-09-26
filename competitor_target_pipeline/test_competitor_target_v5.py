import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import competitor_target_v4 as v4
import competitor_target_v5 as v5
from review_workbench import digest, validate_entry


class ReviewAndVisibilityTests(unittest.TestCase):
    def test_u29_constant_shift_preserves_void_and_interior_spread(self):
        raw=np.full((90,150),60.,np.float32)
        labels=np.zeros(raw.shape,np.int32)
        labels[20:65,15:60]=1;labels[20:65,90:135]=2
        raw[labels==1]=85;raw[labels==2]=130
        raw[35:40,30:35]=60
        field=v5.outer_feather_correction(raw,labels,[(1,15,20,45,45),(2,90,20,45,45)])
        self.assertLess(np.std(field[labels==1]),1e-5)
        self.assertAlmostEqual(float((raw+field)[25,30]-(raw+field)[36,31]),25.,places=4)
        self.assertLessEqual(float(np.abs(field).max()),12.)
        self.assertGreater(field[19,30],0)

    def test_gain_skips_u30_and_requires_evidence(self):
        raw=np.full((80,100),70.,np.float32)
        labels=np.zeros(raw.shape,np.int32)
        for component in ['U30','U29','J1']:
            result,gain,_=v5.local_gain(raw,labels,[],2.,component)
            np.testing.assert_array_equal(result,raw)
            np.testing.assert_array_equal(gain,np.ones_like(raw))

    def test_adaptive_gain_is_bounded(self):
        raw=np.full((260,320),70.,np.float32)
        labels=np.zeros(raw.shape,np.int32)
        regions=[]
        for y in range(20,230,35):
            for x in range(15,300,35):
                i=len(regions)+1;labels[y:y+12,x:x+12]=i
                raw[y:y+12,x:x+12]=80 if 85<y<170 else 105
                regions.append((i,x,y,12,12))
        result,gain,_=v5.local_gain(raw,labels,regions,1.,'J1')
        self.assertGreater(float(gain.max()),1.)
        self.assertLessEqual(float(gain.max()),1.300001)
        self.assertLessEqual(float(np.abs(result-raw).max()),6.00001)

    def test_square_refinement_requires_outer_edge_support(self):
        raw=np.full((140,150),60.,np.float32)
        labels=np.zeros(raw.shape,np.int32);labels[43:77,43:77]=1
        regions=[(1,43,43,34,34)]
        _,_,decisions=v5.refine_squares(raw,labels,regions,1.)
        self.assertFalse(decisions[0]['refined'])
        raw[40:80,40:80]=110
        new,boxes,decisions=v5.refine_squares(raw,labels,regions,1.)
        self.assertTrue(decisions[0]['refined'])
        self.assertGreater(boxes[0][3],34)
        self.assertEqual(new[60,60],1)

    def test_u30_stays_on_v4_processing(self):
        yy,xx=np.mgrid[:100,:140]
        raw=(70+4*np.sin(xx/10)+2*np.cos(yy/7)).astype(np.float32)
        old=v4.process(raw,'U30')[0]['reference_gray']
        new=v5.process_legacy(raw,'U30')[0]['reference_gray']
        np.testing.assert_allclose(old,new,atol=1e-5)

    def test_annotations_require_review_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'sample.png'
            Image.fromarray(np.full((50,50),80,np.uint8)).save(source)
            entry=dict(filename=source.name,shape=[50,50],sha256=digest(source),reviewed=False,reviewer='',annotations=[])
            with self.assertRaises(ValueError):validate_entry(entry,source)
            healthy=dict(kind='healthy',pad_id='1',points=[[2,2],[12,2],[12,12],[2,12]])
            background=dict(kind='background',pad_id='bg',points=[[30,30],[40,30],[40,40],[30,40]])
            entry.update(reviewed=True,reviewer='synthetic-test',annotations=[healthy,background])
            self.assertEqual(len(validate_entry(entry,source)),2)
            entry['annotations'].append(dict(healthy,kind='defect'))
            with self.assertRaises(ValueError):validate_entry(entry,source)
            entry['annotations'].pop();entry['sha256']='bad'
            with self.assertRaises(ValueError):validate_entry(entry,source)


if __name__=='__main__':unittest.main()
