from datetime import datetime, timedelta, date
import cv2
from Utils.timer import Timer
from Utils.smile_result import SmileResult
from Utils.video_manager import VideoManager
import csv
from model_predictor import ModelPredictor, PredictionType
from Feature_Extract.image_processing import crop_face
import pyttsx3
from upload_to_aws import upload_file
from email_manager import Email
from data.voices.voices_database import random_happy_sentence, random_sad_sentence
from keras import backend as K

NUM_OF_SKIP_CAP = 15
SMILE_THRESHOLD = 0.5
face_cascade = cv2.CascadeClassifier("data/haarcascade/haarcascade_frontalface_default.xml")


class VideoMain(object):

    def __init__(self, window_name, is_doll, algo, time, email):
        K.clear_session()
        # Instance Variable
        self.time_of_interval = time
        if email == '':
            self.email = None
        else:
            self.email = Email(email)
        self.is_doll = is_doll
        self.model = ModelPredictor(algo)
        self.engine = pyttsx3.init()
        self.smile_results = []
        self.time_when_start = None
        self.still_running = True
        self.video = VideoManager(window_name, SMILE_THRESHOLD, is_micro_controller=0)
        self.interval_timer = Timer()
        self.smile_timer = Timer()

    def stop_detecting(self):
        self.still_running = False
        self.video.stop_video()
        self.end_process()

    def to_csv(self, csv_file):
        for i in range(0, len(self.smile_results)):
            csv_columns = ['Number of Detected Face', 'Percent Smile', 'High accuracy of smile', 'Max Time of Smile']
            writer = csv.writer(csv_file)
            end_interval_time = self.time_when_start + timedelta(seconds=self.time_of_interval)
            writer.writerow(['****Result between ' + str(self.time_when_start.strftime("%H:%M:%S")) + ' - ' +
                             str(end_interval_time.strftime("%H:%M:%S")) + '*****'])
            writer.writerow(csv_columns)
            writer.writerow([
                self.smile_results[i].get_num_face_detected(),
                self.smile_results[i].get_percentage(),
                self.smile_results[i].get_max_smile(),
                self.smile_results[i].get_max_time_of_smile()
            ])
            self.time_when_start = end_interval_time
            print('****Result between ' + str(self.time_when_start.strftime("%H:%M:%S")) + ' - ' +
                  str(end_interval_time.strftime("%H:%M:%S")) + '*****')
            self.smile_results[i].print_smile_details()

    def end_process(self):
        self.interval_timer.stop()
        file_name = 'Smile Results'+str(date.today())+'.csv'
        with open(file_name, 'w', newline='', encoding='utf8') as f:
            self.to_csv(csv_file=f)
        try:
            if self.email is not None:
                self.email.send_email(file_name)
            upload_file(file_name)
        except:
            print("An exception occurred while uploading to aws")

    @staticmethod
    def initialize_ver_for_report(self):
        num_of_smiles = 0
        num_of_detected_face = 0
        max_time_of_smile, time_of_smile = 0, 0
        max_class_of_smile = 0
        return num_of_smiles, num_of_detected_face, max_time_of_smile, time_of_smile, max_class_of_smile

    @staticmethod
    def analyze_prediction(self, classes, is_smile, max_class_of_smile, num_of_smiles, time_of_smile):
        if classes > 0.50:
            if not is_smile:
                is_smile = True
                self.smile_timer.start()
            if classes > max_class_of_smile:
                max_class_of_smile = classes
            num_of_smiles += 1
        else:
            if is_smile:
                is_smile = False
                time_of_smile = self.smile_timer.get_time()
                self.smile_timer.stop()
        return is_smile, max_class_of_smile, num_of_smiles, time_of_smile

    def make_sound(self, classes):
        if classes[0] > 0.5:
            self.engine.say(random_happy_sentence())
        else:
            self.engine.say(random_sad_sentence())

    def start_detecting(self):
        self.video.start_video()
        last_prediction_is_smile = False
        frame_counter = 0  # to sample every 5 frames
        num_of_smiles, num_of_detected_face, max_time_of_smile, time_of_smile, max_class_of_smile = \
            self.initialize_ver_for_report(self)
        ret, frame = self.video.read_frame()
        self.interval_timer.start()
        is_smile = False
        self.time_when_start = datetime.now()
        while ret:
            if frame_counter % NUM_OF_SKIP_CAP == 0:  # Sends every 5 frame for detection
                ret, frame = self.video.read_frame()
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(1, 1))
                for (x, y, w, h) in faces:
                    if w < 30 and h < 30:  # skip the small faces (probably false detections)
                        continue
                    face_img = crop_face(gray_image, x, y, w, h)
                    last_img = cv2.resize(face_img, (48, 48))
                    data = self.model.get_prediction_data(last_img)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if data is not None:
                        num_of_detected_face += 1
                        classes = self.model.predict(data)
                        print(classes[0])
                        is_smile, max_class_of_smile, num_of_smiles, time_of_smile = \
                            self.analyze_prediction(self, classes[0], is_smile, max_class_of_smile, num_of_smiles,
                                                    time_of_smile)
                        self.video.put_text_on_frame(classes[0], frame, x, y)
                        if self.is_doll:
                            self.make_sound(classes)
                            self.engine.runAndWait()
            if not is_smile:
                max_time_of_smile = time_of_smile if max_time_of_smile < time_of_smile else max_time_of_smile
            if self.interval_timer.get_time() >= self.time_of_interval:
                if is_smile:
                    time_of_smile = self.smile_timer.get_time()
                    max_time_of_smile = time_of_smile if max_time_of_smile < time_of_smile else max_time_of_smile
                    self.smile_timer.stop()
                self.interval_timer.stop()
                percentage_smile = num_of_smiles/num_of_detected_face * 100 if num_of_detected_face > 0 else 0
                self.smile_results.append(SmileResult(max_class_of_smile, percentage_smile, max_time_of_smile,
                                                      num_of_detected_face))
                num_of_smiles, num_of_detected_face, max_time_of_smile, time_of_smile, max_class_of_smile = \
                    self.initialize_ver_for_report(self)
            frame_counter = frame_counter + 1
            self.video.show_video(frame)
            if not self.still_running:
                break



#if __name__ == '__main__':

    #start_detecting(is_doll=True)
