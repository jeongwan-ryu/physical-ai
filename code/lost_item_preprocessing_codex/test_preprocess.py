import unittest
import numpy as np
from preprocess import (Preprocessor, Config, to_linear, to_srgb,
                        tile_labels, merge_detections, LedController, calibrate_gain, flatfield_correct)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(20, 180, (720, 1280, 3), dtype=np.uint8)
        self.p = Preprocessor()

    def test_layout_coverage_and_color(self):
        batch, tiles = self.p.tiles(self.frame[240:])
        self.assertEqual(batch.shape, (2, 3, 480, 672))
        self.assertEqual([t.x for t in tiles], [0, 608])
        np.testing.assert_allclose(batch[0, :, 0, 0], self.frame[240, 0, ::-1]/255, atol=1e-7)
        self.assertTrue(batch.flags.c_contiguous)

    def test_srgb_round_trip_and_identity_gain(self):
        np.testing.assert_array_equal(to_srgb(to_linear(self.frame)), self.frame)
        p = Preprocessor(gain=np.ones((480,1280,1), np.float32))
        np.testing.assert_array_equal(p.correct(self.frame, True), p.correct(self.frame, False))

    def test_label_and_detection_coordinates(self):
        _, tiles = self.p.tiles(self.frame[240:])
        labels, ignored = tile_labels([[0,620,260,640,280], [0,650,300,653,303]], tiles[1])
        self.assertEqual(labels.shape, (1,5))
        self.assertEqual(ignored.shape, (1,4))
        rows = [np.array([[620,20,640,40,.9,0]]), np.array([[12,20,32,40,.8,0]])]
        merged = merge_detections(rows, tiles)
        self.assertEqual(merged.shape, (1,6))
        np.testing.assert_allclose(merged[0,:4], [620,260,640,280])

    def test_padding_and_full_coverage(self):
        p = Preprocessor(Config(roi_top=0))
        _, small = p.tiles(np.zeros((90,100,3),np.uint8))
        self.assertEqual((small[0].valid_height,small[0].valid_width),(90,100))
        self.assertEqual(int(small[0].image[-1,-1,0]),114)
        _, tiles = p.tiles(np.zeros((900,1500,3),np.uint8))
        coverage = np.zeros((900,1500),bool)
        for t in tiles:
            coverage[t.y:t.y+t.valid_height,t.x:t.x+t.valid_width]=True
        self.assertTrue(coverage.all())

    def test_mask_denominator_and_probe(self):
        mask = np.ones((480,1280),bool)
        mask[:, :128] = False
        p = Preprocessor(mask=mask)
        frame = self.frame.copy()
        frame[240:, :128] = 255
        q = p.quality(frame, True)
        self.assertEqual(q['clip_fraction'],1)
        self.assertEqual(q['reason'],'overexposed')
        self.assertIsNone(p.run(frame, probe_frame=True)[0])

    def test_led_probe_uses_actual_metadata(self):
        led = LedController()
        for t in range(5):
            led.update(30,4,t)
        self.assertTrue(led.on)
        self.assertTrue(led.request_probe(6))
        led.update(4,1,6)  # Ordinary fixed-exposure LED frame cannot turn LED off.
        self.assertTrue(led.on)
        led.update(10,4,7,probe_frame=True)
        self.assertFalse(led.on)

    def test_flatfield_validation(self):
        gain = calibrate_gain([np.full((720,1280,3),128,np.uint8)])
        np.testing.assert_allclose(gain,1)
        with self.assertRaises(ValueError):
            Preprocessor(gain=np.zeros((480,1280)))


class RaspberryTests(unittest.TestCase):
    def test_cpu_lookup_matches_reference(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0,256,(80,128,3),dtype=np.uint8)
        gain = rng.uniform(.2,2.5,(80,128,1)).astype(np.float32)
        actual = flatfield_correct(frame, gain)
        expected = to_srgb(to_linear(frame) * gain)
        self.assertLessEqual(int(np.abs(actual.astype(int)-expected.astype(int)).max()),1)

    def test_csi_metadata_and_request_release(self):
        from raspberry_camera import CsiCamera
        class Request:
            released = False
            def make_array(self, name):
                return np.array([[[1,2,3]]],np.uint8)
            def get_metadata(self):
                return {'ExposureTime':4000,'AnalogueGain':2,'SensorTimestamp':123}
            def release(self):
                self.released = True
        request = Request()
        class Camera:
            def capture_request(self):
                return request
        adapter = CsiCamera.__new__(CsiCamera)
        adapter.camera = Camera()
        frame, metadata = adapter.read()
        self.assertTrue(request.released)
        self.assertEqual(metadata['exposure_ms'],4)
        self.assertEqual(metadata['analog_gain'],2)
        np.testing.assert_array_equal(frame,[[[1,2,3]]])


if __name__ == '__main__':
    unittest.main()
