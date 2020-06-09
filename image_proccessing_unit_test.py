import unittest
from Feature_Extract import image_processing
import cv2

happyChildAllProccess = cv2.cvtColor(cv2.imread('data/testing images/happyChildReal.jpg'), cv2.COLOR_RGB2GRAY)

class ModelTest(unittest.TestCase):

    def testOurReshapeImageToSquare(self):
        self.assertNotEqual(happyChildAllProccess.shape[0], 48)
        self.assertNotEqual(happyChildAllProccess.shape[1], 48)
        new_image = image_processing.image_resize_to_square(happyChildAllProccess, 64, 64)
        height = new_image.shape[0]
        width = new_image.shape[1]
        self.assertEqual(height, 64)
        self.assertEqual(width, 64)

    def testOurReshapeImageToSquareFailesWithNotEqualMaseurment(self):
        def test(self):
            with self.assertRaises(Exception) as context:
                image_processing.image_resize_to_square(happyChildAllProccess, 20, 64)
            self.assertTrue('This is broken' in context.exception)


    def testOurReshapeImage(self):
        self.assertNotEqual(happyChildAllProccess.shape[0], 24)
        self.assertNotEqual(happyChildAllProccess.shape[1], 32)
        new_image = image_processing.image_resize(happyChildAllProccess, 24, 32)
        height = new_image.shape[0]
        width = new_image.shape[1]
        self.assertEqual(height, 24)
        self.assertEqual(width, 32)


if __name__ == '__main__':
    unittest.main()






