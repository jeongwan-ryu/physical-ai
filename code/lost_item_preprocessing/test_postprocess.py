import unittest
import numpy as np
from preprocess import Preprocessor
from postprocess import LostItemPostprocessor, PostConfig, MapConfirmer


class PostTests(unittest.TestCase):
    def setUp(self):
        _, self.tiles = Preprocessor().tiles(np.zeros((480,1280,3),np.uint8))

    def test_merge_reflection_classes_sizes(self):
        mask = np.zeros((480,1280),bool)
        mask[20:40,620:640] = True
        p = LostItemPostprocessor(PostConfig(target_classes=(0,1),thresholds={0:.35},
                                             min_sizes={1:10}),mask)
        a = [[620,20,640,40,.5,0], [20,20,26,26,.9,1], [50,50,70,70,.9,2]]
        b = [[12,20,32,40,.8,0]]
        out = p.process([a,b], self.tiles, led_on=True)
        self.assertEqual(out.shape,(1,6))
        np.testing.assert_allclose(out[0,:4],[620,260,640,280])
        self.assertAlmostEqual(float(out[0,4]),.8)

    def test_yolo_adapter(self):
        class Boxes:
            xyxy=np.array([[10,20,30,40]])
            conf=np.array([.9])
            cls=np.array([0])
        class Result:
            orig_shape=(480,672)
            boxes=Boxes()
        out = LostItemPostprocessor().from_yolo([Result(),Result()],self.tiles)
        self.assertEqual(out.shape,(2,6))
        self.assertEqual(out[1,0],618)

    def test_confirmation_distinct_frames_and_expiry(self):
        c = MapConfirmer()
        repeated=[[0,1,2,.9],[0,1,2,.8]]
        self.assertEqual(len(c.update(0,repeated)),0)
        self.assertEqual(len(c.update(1,[])),0)
        self.assertEqual(len(c.update(2,repeated)),0)
        self.assertEqual(len(c.update(3,[[0,1.01,2,.9]])),1)
        self.assertEqual(len(c.update(10,repeated)),0)
        with self.assertRaises(ValueError):
            c.update(10,repeated)

    def test_class_separation_and_mask_validation(self):
        c=MapConfirmer(need=2)
        c.update(0,[[0,1,2,.9]])
        self.assertEqual(len(c.update(1,[[1,1,2,.9]])),0)
        p=LostItemPostprocessor(reflection_mask=np.zeros((1,1),bool))
        with self.assertRaises(ValueError):
            p.process([[],[]],self.tiles)


if __name__ == '__main__':
    unittest.main()
