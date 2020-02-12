import dlib
import numpy as np
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
data = {}


def get_landmarks(image):
    eyes_and_mouth_points = []
    #image = cv2.imread(image_path)
    detections = detector(image, 1)
    if len(detections) < 1:
        return
    for k, d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0, 68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in ylist]
        i = 1
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)
            if (i > 36):
                eyes_and_mouth_points.append(dist)
                eyes_and_mouth_points.append((math.atan2(y, x)*360)/(2*math.pi))
            i = i + 1
    return eyes_and_mouth_points
