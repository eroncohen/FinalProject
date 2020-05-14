import cv2
from imutils.video import VideoStream


class VideoError(Exception):
    """A custom exception used to report errors in use of Video Manager class"""


class VideoManager:
    def _init_(self, window_name, smile_threshold, is_micro_controller):
        self.window_name = window_name
        self.smile_threshold = smile_threshold
        self.is_micro_controller = is_micro_controller
        self.cap = None

    def start_video(self):
        if self.window_name is None:
            raise VideoError(f"No window name for video")
        cv2.namedWindow(self.window_name)
        if self.is_micro_controller == 1:
            self.cap = VideoStream(src=0).start()
            return True, self.cap
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            return self.cap

    def stop_video(self):
        cv2.destroyWindow(self.window_name)
        self.cap.release()

    def show_video(self, frame):
        cv2.imshow(self.window_name, frame)

    def read_frame(self):
        if self.is_micro_controller == 0:
            if not self.cap.isOpened():
                raise VideoError(f"No video available to read")
                return False, False
        if self.is_micro_controller == 1:
            return True, self.cap.read()
        return self.cap.read()

    def put_text_on_frame(self, classes, frame, x, y):
        cv2.putText(frame, "Not Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) if classes < self.smile_threshold else cv2.putText(frame, "Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)