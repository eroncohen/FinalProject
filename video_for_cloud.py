from image_analyzer import FaceCropper
import cv2
import requests


addr = 'http://54.161.173.40:5000/'
test_url = addr + 'predict'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


def crop_video_and_predict_result(is_cloud, video_path):
    window_name = "Live Video Feed"
    cv2.namedWindow(window_name)
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    if not is_cloud:
        detector = FaceCropper()
    frame_counter = 0  # to sample every 5 frames
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    print(ret)
    while ret:
        if frame_counter % 5 == 0:  # Sends every 5 frame for detection
            ret, frame = cap.read()
            if is_cloud:
                # encode image as jpeg
                _, img_encoded = cv2.imencode('.jpg', frame)
                # send http request with image and receive response
                response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
                print(response.content)
            else:
                # local detect
                detector.generate(frame)
                # cloud detect

        frame_counter = frame_counter + 1
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyWindow(window_name)
    cap.release()


if __name__ == '__main__':
    crop_video_and_predict_result(is_cloud=True, video_path=None)

