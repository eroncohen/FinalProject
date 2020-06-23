import unittest
from Utils.model_manager import ModelManager, PredictionType
from Feature_Extract.image_processing import crop_face
import cv2

happyMan = cv2.cvtColor(cv2.imread('../data/testing images/happyMan.jpeg'), cv2.COLOR_RGB2GRAY)
naturalMan = cv2.cvtColor(cv2.imread('../data/testing images/naturalMan.jpeg'), cv2.COLOR_RGB2GRAY)
happyWomen = cv2.cvtColor(cv2.imread('../data/testing images/happyWomen.jpeg'), cv2.COLOR_RGB2GRAY)
naturalWomen = cv2.cvtColor(cv2.imread('../data/testing images/naturalWomen.jpeg'), cv2.COLOR_RGB2GRAY)

happyChildAllProccess = cv2.imread('../data/testing images/happyChildReal.jpg')

modelCNN = ModelManager(PredictionType.CNN)
modelYE = ModelManager(PredictionType.YE_ALGORITHM)


class ModelTest(unittest.TestCase):

    def testChildSmileAllProccessCNN(self):
        gray_image = cv2.cvtColor(happyChildAllProccess, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml").detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))
        for (x, y, w, h) in faces:
            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                continue
            face_img = crop_face(gray_image, x, y, w, h)
            last_img = cv2.resize(face_img, (48, 48))
            data = modelCNN.get_prediction_data(last_img)
            self.assertGreater(modelCNN.predict(data)[0], 0.9)

    def testChildSmileAllProccessYE(self):
        gray_image = cv2.cvtColor(happyChildAllProccess, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml").detectMultiScale(
            gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))
        for (x, y, w, h) in faces:
            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                continue
            face_img = crop_face(gray_image, x, y, w, h)
            last_img = cv2.resize(face_img, (48, 48))
            data = modelYE.get_prediction_data(last_img)
            self.assertGreater(modelYE.predict(data), 0.9)

    def testSmileManModelCNN(self):
        self.assertGreater(modelCNN.predict(happyMan)[0], 0.9)

    def testSmileManModelCNN(self):
        self.assertGreater(modelCNN.predict(happyMan)[0], 0.7)

    def testSmileWomenModelCNN(self):
        self.assertGreater(modelCNN.predict(happyWomen)[0], 0.9)

    def testNaturalWomenModelCNN(self):
        self.assertLess(modelCNN.predict(naturalWomen)[0], 0.2)

    def testNaturalMannModelCNN(self):
        self.assertLess(modelCNN.predict(naturalMan)[0], 0.2)

    def testSmileManModelYE(self):
        self.assertGreater(modelYE.predict(modelYE.get_prediction_data(happyMan)), 0.7)

    def testSmileWomenModelYE(self):
        self.assertGreater(modelYE.predict(modelYE.get_prediction_data(happyWomen)), 0.7)

    def testNaturalWomenModelYE(self):
        self.assertLess(modelYE.predict(modelYE.get_prediction_data(naturalWomen)), 0.5)

    def testNaturalMannModelYE(self):
        self.assertLess(modelYE.predict(modelYE.get_prediction_data(naturalMan)), 0.5)


if __name__ == '__main__':
    unittest.main()






