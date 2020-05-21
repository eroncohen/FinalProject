import cv2
from model_predictor import load_model_func, pred


def crop_face(gray_image, x, y, w, h):
    r = max(w, h) / 2
    center_x = x + w / 2
    center_y = y + h / 2
    nx = int(center_x - r)
    ny = int(center_y - r)
    nr = int(r * 2)
    return gray_image[ny:ny + nr, nx:nx + nr]


class FaceCropper(object):
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        self.loaded_model = load_model_func('model_cnn.json','weights_cnn.h5')

    def generate(self, frame):
        '''
        This Function generate the frame by searching for face inside it.
        If there is frame sends the crop face for prediction

        The prediction comes back as a fraction between 0-1. The closest to 1 is the the closet the model
        detected a face
        '''
        if frame is None:
            print("Can't open image file")
            return 0

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))
        classes = None
        for (x, y, w, h) in faces:
            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                continue
            face_img = crop_face(gray_image, x, y, w, h)
            last_img = cv2.resize(face_img, (48, 48))
            classes = pred(last_img, self.loaded_model)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if classes > 0.8:
                cv2.putText(frame, "Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(classes)
        if classes:
            return classes
        else:
            return [["No face found"]]




