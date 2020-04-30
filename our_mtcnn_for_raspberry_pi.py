from datetime import datetime, timedelta
from extract_fetures_image_proc import our_mtcnn
import cv2
import numpy as np
from joblib import load
from timer import Timer
from smile_result import SmileResult
from imutils.video import VideoStream
import time
from video_manager import VideoManager
from imutils.video import FPS
import csv


NUM_OF_SKIP_CAP = 5
TIME_OF_INTERVAL = 5
WINDOW_NAME = 'Smile Machine'
SMILE_THRESHOLD = 0.5
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
model = load('svm_model_our_mtcnn_new2.joblib')
interval_timer = Timer()
smile_timer = Timer()
video = VideoManager(WINDOW_NAME, SMILE_THRESHOLD)


def crop_face(gray_image, x, y, w, h):
    r = max(w, h) / 2
    center_x = x + w / 2
    center_y = y + h / 2
    nx = int(center_x - r)
    ny = int(center_y - r)
    nr = int(r * 2)
    return gray_image[ny:ny + nr, nx:nx + nr]


def to_csv(csv_file, smile_results, time_when_start):
    for i in range(0, len(smile_results)):
        csv_columns = ['Number of Detected Face', 'Percent Smile', 'High accuracy of smile', 'Max Time of Smile']
        writer = csv.writer(csv_file)
        end_interval_time = time_when_start + timedelta(seconds=TIME_OF_INTERVAL)
        writer.writerow(['********Result between ' + str(time_when_start.strftime("%H:%M:%S")) + ' - ' +  str(end_interval_time.strftime("%H:%M:%S")) + '*************'])
        writer.writerow(csv_columns)
        writer.writerow([
            smile_results[i].get_num_face_detected(),
            smile_results[i].get_percentage(),
            smile_results[i].get_max_smile(),
            smile_results[i].get_max_time_of_smile()
        ])
        time_when_start = end_interval_time
        print('********Result between ' + str(time_when_start.strftime("%H:%M:%S")) + ' - ' +  str(end_interval_time.strftime("%H:%M:%S")) + '*************')
        smile_results[i].print_smile_details()


def end_process(smile_results, time_when_start):
    with open('Smile Results.csv', 'w', newline='', encoding='utf8') as f:
        to_csv(f, smile_results, time_when_start)


def initialize_ver_for_report():
    num_of_smiles = 0
    num_of_detected_face = 0
    max_time_of_smile, time_of_smile = 0, 0
    max_class_of_smile = 0
    return num_of_smiles, num_of_detected_face, max_time_of_smile, time_of_smile, max_class_of_smile


def analyze_prediction(classes, is_smile, max_class_of_smile, num_of_smiles, time_of_smile):
    if classes > 0.50:
        if not is_smile:
            is_smile = True
            smile_timer.start()
        if classes > max_class_of_smile:
            max_class_of_smile = classes
        num_of_smiles += 1
    else:
        if is_smile:
            is_smile = False
            time_of_smile = smile_timer.get_time()
            smile_timer.stop()
    return is_smile, max_class_of_smile, num_of_smiles, time_of_smile


def start_detecting():
    smile_results = []
    frame_counter = 0  # to sample every 5 frames
    num_of_smiles, num_of_detected_face, max_time_of_smile, time_of_smile, max_class_of_smile = initialize_ver_for_report()
    fps = FPS().start()
    interval_timer.start()
    is_smile = False
    time_when_start = datetime.now()
    while True:
        if frame_counter % NUM_OF_SKIP_CAP == 0:  # Sends every 5 frame for detection
            frame = vs.read()
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
                    num_of_detected_face += 1
                    new_arr = np.array([feature_vector])
                    classes = model.predict_proba(new_arr[0:1])[:, 1]
                    print(classes[0])
                    is_smile, max_class_of_smile, num_of_smiles, time_of_smile = analyze_prediction(classes[0], is_smile, max_class_of_smile, num_of_smiles, time_of_smile)
                    video.put_text_on_frame(classes[0], frame, x, y)
        if not is_smile:
            if max_time_of_smile < time_of_smile:
                max_time_of_smile = time_of_smile
        if interval_timer.get_time() >= TIME_OF_INTERVAL:
            if is_smile:
                time_of_smile = smile_timer.get_time()
                if max_time_of_smile < time_of_smile:
                    max_time_of_smile = time_of_smile
                smile_timer.stop()
            interval_timer.stop()
            percentage_smile = num_of_smiles/num_of_detected_face * 100 if num_of_detected_face > 0 else 0
            smile_results.append(SmileResult(max_class_of_smile, percentage_smile, max_time_of_smile, num_of_detected_face))
            num_of_smiles, num_of_detected_face, max_time_of_smile, time_of_smile, max_class_of_smile = initialize_ver_for_report()
        frame_counter = frame_counter + 1
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
    #video.stop_video()
    end_process(smile_results, time_when_start)


if __name__ == "__main__":
    #video.start_video()
    vs = VideoStream(src=0).start()
    #time.sleep(2.0)
    start_detecting()
