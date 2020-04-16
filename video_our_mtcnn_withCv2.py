from binary_image import crop_mouth_from_face
from resize_image import image_resize
from resize_image import image_resize_to_square
from sklearn.preprocessing import MinMaxScaler
from extract_fetures_image_proc import our_mtcnn
import cv2
import numpy as np
from joblib import load
from timer import Timer

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
model = load('svm_model_our_mtcnn_new2.joblib')
t = Timer()
smile_timer = Timer()

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
num_of_smiles = 0
max_time_of_smile, time_of_smile = 0, 0
max_class_of_smile = 0
t.start()
is_smile = False
while ret:
    if frame_counter % 5 == 0:  # Sends every 5 frame for detection
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))
        for (x, y, w, h) in faces:
            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                continue
            face_img = crop_face(gray_image, x, y, w, h)
            last_img = cv2.resize(face_img, (48, 48))

            feature_vector = our_mtcnn(image_path=None, image=last_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if feature_vector is not None:
                new_arr = np.array([feature_vector])
                classes = model.predict_proba(new_arr[0:1])[:, 1]
                print(classes)
                if classes[0] > 0.50:
                    if is_smile:
                        is_smile = True
                        smile_timer.start()
                    if classes[0] > max_class_of_smile:
                        max_class_of_smile = classes[0]
                    cv2.putText(frame, "Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    num_of_smiles += 1
                else:
                    if is_smile:
                        is_smile = False
                        time_of_smile = smile_timer.get_time()
                        smile_timer.stop()
                    cv2.putText(frame, "Not Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if not is_smile:
        if max_time_of_smile < time_of_smile:
            max_time_of_smile = time_of_smile
    if t.get_time() >= 30:
        t.stop()
        percentage_smile = num_of_smiles/(frame_counter/5) * 100
        print('% ' + str(percentage_smile))
        print('longest smile ' + str(max_time_of_smile))
        print('max class' + str(max_class_of_smile))

    frame_counter = frame_counter + 1
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow(window_name)
cap.release()
