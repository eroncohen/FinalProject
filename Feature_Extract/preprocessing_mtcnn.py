import cv2
import numpy as np
import math
from Feature_Extract.extract_fetures_image_proc import ye_algorithm_detect_five_points


def get_five_points_distance_and_angle(image, is_ye_algorithm):
    if is_ye_algorithm:
        points_list = ye_algorithm_detect_five_points(None, image)
        if points_list is None:
            return
    else:
        from mtcnn import MTCNN
        detector = MTCNN()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json = detector.detect_faces(img)
        if len(json) == 0:
            return
        key_points_json = json[0]['keypoints']
        points_list = []
        points_list.append(key_points_json['left_eye'])
        points_list.append(key_points_json['right_eye'])
        points_list.append(key_points_json['nose'])
        points_list.append(key_points_json['mouth_left'])
        points_list.append(key_points_json['mouth_right'])
    x_list = []
    y_list = []
    for i in range(0, 5):  # Store X and Y coordinates in two lists
        x_list.append(float(points_list[i][0]))
        y_list.append(float(points_list[i][1]))
    x_mean = np.mean(x_list)  # Find both coordinates of centre of gravity
    y_mean = np.mean(y_list)

    x_central = [(x - x_mean) for x in x_list]  # Calculate distance centre <-> other points in both axes
    y_central = [(y - y_mean) for y in y_list]
    landmarks_vectorised = []
    i = 1
    for x, y, w, z in zip(x_central, y_central, x_list, y_list):
        mean_np = np.asarray((y_mean, x_mean))
        coor_np = np.asarray((z, w))
        dist = np.linalg.norm(coor_np - mean_np)
        landmarks_vectorised.append(dist)
        landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
        i = i + 1
    return landmarks_vectorised
