import unittest
import numpy as np
from process_bubble import process_bubble

class TestProcessBubble(unittest.TestCase):

    def test_no_contours(self):
        # Create a black image that will not produce any contours
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # This call should not raise a ValueError
        try:
            processed_image, contour = process_bubble(black_image)
            # Check that the returned image is the same as the input
            self.assertTrue(np.array_equal(processed_image, black_image))
            # Check that the contour is None
            self.assertIsNone(contour)
        except ValueError:
            self.fail("process_bubble() raised ValueError unexpectedly!")

if __name__ == '__main__':
    unittest.main()
