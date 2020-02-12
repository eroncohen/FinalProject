import cv2
from preprocessing_mtcnn import get_landmarks
#from preprocessing_dlib import get_landmarks
import numpy as np
from joblib import load

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
model = load('svm_modelMTCNN2.joblib')


def crop_face(gray_image, x, y, w, h):
    r = max(w, h) / 2
    center_x = x + w / 2
    center_y = y + h / 2
    nx = int(center_x - r)
    ny = int(center_y - r)
    nr = int(r * 2)
    return gray_image[ny:ny + nr, nx:nx + nr]


window_name = "Live Video Feed"
cv2.namedWindow(window_name)
cap = cv2.VideoCapture(0)
frame_counter = 0  # to sample every 5 frames
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
while ret:

    if frame_counter % 15 == 0:  # Sends every 5 frame for detection
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))

        for (x, y, w, h) in faces:
            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                continue
            face_img = crop_face(gray_image, x, y, w, h)
            last_img = cv2.resize(face_img, (48, 48))
            cv2.imshow("image", last_img)
            arr = get_landmarks(last_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if arr is not None:
                new_arr = np.array([arr])
                classes = model.predict_proba(new_arr[0:1])[:, 1]
                print(classes)
                if classes[0] > 0.75:
                    cv2.putText(frame, "Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Not Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    frame_counter = frame_counter + 1
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow(window_name)
cap.release()

