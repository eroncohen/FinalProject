from binary_image import crop_mouth_from_face
from resize_image import image_resize
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
from joblib import load

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
model = load('svm_model_mouth2.joblib')

def scaling(X_train):
    preproc = MinMaxScaler()
    return preproc.fit_transform([X_train])


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
    if frame_counter % 20 == 0:  # Sends every 5 frame for detection
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))
        for (x, y, w, h) in faces:
            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                continue
            face_img = crop_face(gray_image, x, y, w, h)
            last_img = cv2.resize(face_img, (48, 48))

            new_mouth_image = crop_mouth_from_face(last_img)
            #scaled_pic = scaling(new_mouth_image)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            classes = model.predict_proba(new_mouth_image.reshape(1,-1))[:, 1]
            print(classes)

            if classes[0] > 0.50:
                cv2.putText(frame, "Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    frame_counter = frame_counter + 1
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow(window_name)
cap.release()

'''
 new_image = image_resize(gray_image, 48, 48)
        new_mouth_image = crop_mouth_from_face(new_image)
        scaled_pic = scaling(new_mouth_image)

        classes = model.predict_proba(scaled_pic)[:, 1]
        print(classes[0])
    frame_counter = frame_counter + 1
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow(window_name)
cap.release()

'''

