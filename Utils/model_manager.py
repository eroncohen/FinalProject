from enum import Enum
from joblib import load
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json


class PredictionType(Enum):
    CNN = 0
    DLIB = 1
    MTCNN = 2
    MOUTH_CNN = 3
    MOUTH_VECTOR = 4
    YE_ALGORITHM = 5


class ModelManager:
    def __init__(self, prediction_type):
        if not isinstance(prediction_type, PredictionType):
            raise TypeError('prediction_type must be an instance of PredictionType Enum')
        self.prediction_type = prediction_type
        if self.prediction_type == PredictionType.CNN:
            self.model = load_model_func('../Models/Classifiers/model_cnn.json', 'Models/weights/weights_cnn.h5')
        elif self.prediction_type == PredictionType.DLIB:
            self.model = load("../Models/Classifiers/svm_model_dlib1.joblib")
        elif self.prediction_type == PredictionType.MTCNN:
            from Feature_Extract.preprocessing_5points import init_detector_mtcnn
            init_detector_mtcnn()
            self.model = load("../Models/Classifiers/svm_modelMTCNN2.joblib")
        elif self.prediction_type == PredictionType.MOUTH_CNN:
            self.model = load_model_func('../Models/Classifiers/model_mouth_cnn.json', 'Models/weights/weights_mouth_cnn.h5')
        elif self.prediction_type == PredictionType.MOUTH_VECTOR:
            self.model = load('../Models/Classifiers/svm_model_mouth_vector0.joblib')
        elif self.prediction_type == PredictionType.YE_ALGORITHM:
            self.model = load('../Models/Classifiers/svm_model_our_mtcnn.joblib')

    def get_prediction_data(self, img):
        from Feature_Extract.image_processing import crop_mouth_from_face
        from Feature_Extract.preprocessing_5points import get_five_points_distance_and_angle

        if self.prediction_type == PredictionType.CNN:
            data = img

        elif self.prediction_type == PredictionType.DLIB:
            from Feature_Extract.preprocessing_dlib import get_landmarks_dlib
            facial_landmarks = get_landmarks_dlib(img)
            data = self.get_data_as_numpy_array(facial_landmarks)

        elif self.prediction_type == PredictionType.MTCNN:
            facial_landmarks = get_five_points_distance_and_angle(img, is_ye_algorithm=False)
            data = self.get_data_as_numpy_array(facial_landmarks)

        elif self.prediction_type == PredictionType.MOUTH_CNN:
            data = crop_mouth_from_face(img, is_cnn=True)

        elif self.prediction_type == PredictionType.MOUTH_VECTOR:
            data = crop_mouth_from_face(img, is_cnn=False)

        elif self.prediction_type == PredictionType.YE_ALGORITHM:
            feature_vector = get_five_points_distance_and_angle(img, is_ye_algorithm=True)
            data = self.get_data_as_numpy_array(feature_vector)

        return data

    def get_data_as_numpy_array(self, feature_vector):
        return np.array([feature_vector]) if feature_vector is not None else None

    def predict(self, data):
        if self.prediction_type == PredictionType.CNN or self.prediction_type == PredictionType.MOUTH_CNN:
            return pred(data, self.model)
        else:
            return self.model.predict_proba([data])[:, 1] if self.prediction_type == PredictionType.MOUTH_VECTOR else \
                self.model.predict_proba(data[0:1])[:, 1]


def load_model_func(model_path, weights_path):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    print(loaded_model.summary())
    return loaded_model


def pred(img, loaded_model):
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    single_image = np.vstack([image_array])
    prediction_class = loaded_model.predict(single_image)
    return prediction_class
