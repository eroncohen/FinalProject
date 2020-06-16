from sklearn.preprocessing import MinMaxScaler
import Feature_Extract.image_processing as ip
import cv2
import numpy as np
from joblib import load

model = load('Models/svm_model_mouth2.joblib')


def scaling(x_train):
    preproc = MinMaxScaler()
    return preproc.fit_transform([x_train])


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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_counter = 0  # to sample every 5 frames
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
while ret:
    ret, frame = cap.read()
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow(window_name)
cap.release()


