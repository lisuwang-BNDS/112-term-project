import unittest

from eye_tracker import _clamp, _scale_point_to_screen


class EyeTrackerTests(unittest.TestCase):
    def test_clamp_rounds_to_bounds(self):
        self.assertEqual(_clamp(5, 0, 10), 5)
        self.assertEqual(_clamp(-1, 0, 10), 0)
        self.assertEqual(_clamp(11, 0, 10), 10)

    def test_scale_point_to_screen(self):
        self.assertEqual(_scale_point_to_screen(0, 0, 100, 200, 400, 300), (20, 20))
        self.assertEqual(_scale_point_to_screen(50, 100, 100, 200, 400, 300), (200, 150))
        self.assertEqual(_scale_point_to_screen(150, 50, 100, 200, 400, 300), (380, 75))


if __name__ == '__main__':
    unittest.main()
