import numpy as np
import math
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
data = {}


def get_landmarks(image_path):
    image = cv2.imread(image_path)
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0, 68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        print(xmean)
        ymean = np.mean(ylist)
        print(ymean)
        xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        i = 1
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            #landmarks_vectorised.append(w)
            landmarks_vectorised.append(i)
            i = i + 1
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
        print(i)
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


if __name__ == "__main__":
    get_landmarks("C:/Users/Eron/Desktop/PrivateTest/3/PrivateTest_119017.jpg")
    #print(len(data['landmarks_vectorised']))
    print(data)
