from mtcnn import MTCNN
import cv2
import numpy as np
import math

img = cv2.cvtColor(cv2.imread("C:/Users/Eron/PycharmProjects/Final_Project/Training/3/Training_188659.jpg"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread("C:/Users/Eron/Desktop/study/FinalProject/pics/neu.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()


def get_landmarks(image):
    #image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    json = detector.detect_faces(image)
    if len(json) == 0:
        return
    key_points_json = json[0]['keypoints']
    points_list = []
    points_list.append(key_points_json['left_eye'])
    points_list.append(key_points_json['right_eye'])
    points_list.append(key_points_json['nose'])
    points_list.append(key_points_json['mouth_left'])
    points_list.append(key_points_json['mouth_right'])
    xlist = []
    ylist = []
    for i in range(0, 5): #Store X and Y coordinates in two lists

        xlist.append(float(points_list[i][0]))
        ylist.append(float(points_list[i][1]))
    xmean = np.mean(xlist) #Find both coordinates of centre of gravity
    ymean = np.mean(ylist)

    xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
    ycentral = [(y-ymean) for y in ylist]
    landmarks_vectorised = []
    i = 1
    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
        meannp = np.asarray((ymean, xmean))
        coornp = np.asarray((z, w))
        dist = np.linalg.norm(coornp-meannp)
        landmarks_vectorised.append(dist)
        landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        i = i + 1
    return landmarks_vectorised
