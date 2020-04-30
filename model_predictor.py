from enum import Enum
from joblib import load
import numpy as np
from model_cnn import load_model_func, pred


class PredictionType(Enum):
    CNN = 0
    DLIB = 1
    MTCNN = 2
    MOUTH_CNN = 3
    MOUTH_VECTOR = 4
    YE_ALGORITHM = 5


class ModelPredictor:
    def __init__(self, prediction_type):
        if not isinstance(prediction_type, PredictionType):
            raise TypeError('prediction_type must be an instance of PredictionType Enum')
        self.prediction_type = prediction_type
        if self.prediction_type == PredictionType.CNN:
            None
        elif self.prediction_type == PredictionType.DLIB:
            None
        elif self.prediction_type == PredictionType.MTCNN:
            self.model = load("Models/svm_modelMTCNN2.joblib")
        elif self.prediction_type == PredictionType.MOUTH_CNN:
            self.model = load_model_func()
        elif self.prediction_type == PredictionType.MOUTH_VECTOR:
            self.model = load('Models/svm_model_mouth2.joblib')
        elif self.prediction_type == PredictionType.YE_ALGORITHM:
            self.model = load('Models/svm_model_our_mtcnn_new2.joblib')

    def get_prediction_data(self, image):
        from Feature_Extract.image_processing import crop_mouth_from_face
        from Feature_Extract.extract_fetures_image_proc import our_mtcnn
        from Feature_Extract.preprocessing_mtcnn import get_landmarks_mtcnn
        #from Feature_Extract.preprocessing_dlib import get_landmarks_dlib

        if self.prediction_type == PredictionType.CNN:
            None
        #elif self.prediction_type == PredictionType.DLIB:
            #facial_landmarks = get_landmarks_dlib(image)
            #data = self.get_data_as_numpy_array(facial_landmarks)
        elif self.prediction_type == PredictionType.MTCNN:
            facial_landmarks = get_landmarks_mtcnn(image)
            data = self.get_data_as_numpy_array(facial_landmarks)
        elif self.prediction_type == PredictionType.MOUTH_CNN:
            data = crop_mouth_from_face(image, is_cnn=True)
        elif self.prediction_type == PredictionType.MOUTH_VECTOR:
            data = crop_mouth_from_face(image, is_cnn=False)
        elif self.prediction_type == PredictionType.YE_ALGORITHM:
            feature_vector = our_mtcnn(image_path=None, image=image)
            data = self.get_data_as_numpy_array(feature_vector)

        return data

    def get_data_as_numpy_array(self, feature_vector):
        return np.array([feature_vector]) if feature_vector is not None else None

    def predict(self, data):
        if self.prediction_type == PredictionType.CNN or self.prediction_type == PredictionType.MOUTH_CNN:
            return pred(data, self.model)
        else:
            return self.model.predict_proba(data[0:1])[:, 1]
