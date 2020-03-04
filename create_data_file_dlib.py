import os
import csv
import numpy as np
#from preprocessing_dlib import get_landmarks
from preprocessing_mtcnn import get_landmarks
from binary_image import crop_mouth_from_face
from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler()


def scaling(x):
    x.reshape(1, -1)
    return x_scaler.fit_transform(x)


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def create_csv_file():
    csv_file = open('mouth_arr_scaled.csv', 'w', newline='')
    obj = csv.writer(csv_file)
    arr = []

    arr.append("emotion")
    for i in range(0, 225):
        arr.append(i)

    obj.writerow(arr)
    directory_name_list = ['Training', 'PublicTest', 'PrivateTest']
    for name in directory_name_list:
        for x in range(7):
            if x == 3:
                emotion = 1
            else:
                emotion = 0
            directory_name = 'C:/Users/Eron/PycharmProjects/Final_Project/' + name + '/' + str(x)
            directory = os.fsencode(directory_name)

            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                #arr_to_write = get_landmarks(str(directory_name) + '/' + str(filename))
                arr_to_write = crop_mouth_from_face(str(directory_name) + '/' + str(filename))
                scaled_pic = normalize(arr_to_write)
                #print(scaled_pic)
                if scaled_pic is not None:
                    print(str(directory_name) + '/' + str(filename))
                    scaled_pic.put(0, emotion)
                    obj.writerow(scaled_pic)
    csv_file.close()


if __name__ == '__main__':
    create_csv_file()
