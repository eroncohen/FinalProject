import dlib
import numpy as np
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
data = {}
EYE_START = 36
NUM_OF_DLIB_POINTS = 68


def get_landmarks_dlib(image):
    eyes_and_mouth_points = []
    # image = cv2.imread(image_path)
    detections = detector(image, 1)
    if len(detections) < 1:
        return
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        x_list = []
        y_list = []
        for point_num in range(0, NUM_OF_DLIB_POINTS):  # Store X and Y coordinates in two lists
            x_list.append(float(shape.part(point_num).x))
            y_list.append(float(shape.part(point_num).y))
        x_mean = np.mean(x_list)  # Find both coordinates of centre of gravity
        y_mean = np.mean(y_list)
        x_central = [(x-x_mean) for x in x_list]  # Calculate distance centre <-> other points in both axes
        y_central = [(y-y_mean) for y in y_list]
        point_number = 1
        for x, y, w, z in zip(x_central, y_central, x_list, y_list):
            mean_np = np.asarray((y_mean, x_mean))
            coor_np = np.asarray((z, w))
            dist = np.linalg.norm(coor_np-mean_np)
            if point_number > EYE_START:
                eyes_and_mouth_points.append(dist)
                eyes_and_mouth_points.append((math.atan2(y, x)*360)/(2*math.pi))
            point_number = point_number + 1
    return eyes_and_mouth_points
