import cv2

class VideoError(Exception):
    """A custom exception used to report errors in use of Video Manager class"""


class VideoManager:
    def __init__(self, window_name, smile_threshold):
        self.window_name = window_name
        self.smile_threshold = smile_threshold

    def start_video(self):
        if self.window_name is None:
            raise VideoError(f"No window name for video")
        cv2.namedWindow(self.window_name)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        return self.cap

    def stop_video(self):
        cv2.destroyWindow(self.window_name)
        self.cap.release()

    def show_video(self, frame):
        cv2.imshow(self.window_name, frame)

    def read_frame(self):
        if not self.cap.isOpened():
            raise VideoError(f"No video availble to read")
            return False, False

        return self.cap.read()

    def is_cap_open(self):
        return self.cap.isOpened()

    def put_text_on_frame(self, classes, frame, x, y):
        cv2.putText(frame, "Not Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) if classes < self.smile_threshold else cv2.putText(frame, "Happy", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


