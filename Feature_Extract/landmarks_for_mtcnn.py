from mtcnn import MTCNN
import cv2
import numpy as np

detector = MTCNN()

def get_landmarks(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
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
    xmean = int(np.mean(xlist)) #Find both coordinates of centre of gravity
    ymean = int(np.mean(ylist))
    image_analyze(image, xmean, ymean, points_list)


def image_analyze(image, x_mean,y_mean, points_array):
    cv2.circle(image, (x_mean, y_mean), 5, (255, 0, 0) , -1)
    for i in range(len(points_array)):
        cv2.circle(image, (points_array[i][0], points_array[i][1]), 4, (0, 255, 0), -1)
        cv2.line(image, (points_array[i][0], points_array[i][1]), (x_mean, y_mean), (255,255,255), 1)
    image = cv2.resize(image, (800, 800))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    get_landmarks('happy.jpg')
