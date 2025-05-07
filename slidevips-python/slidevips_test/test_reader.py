import unittest
from slidevips import SlideVips


class TestSlideVips(unittest.TestCase):
    def setUp(self):
        self.slide = SlideVips('test_slide.vips', mode='RGB')

    def test_initialization(self):
        self.assertIsInstance(self.slide, SlideVips)
        self.assertEqual(self.slide.mode, 'RGB')

    def test_read_region(self):
        # Assuming read_region takes two arguments: level and coordinates
        region = self.slide.read_region(0, (0, 0, 100, 100))
        self.assertIsNotNone(region)


if __name__ == '__main__':
    unittest.main()
