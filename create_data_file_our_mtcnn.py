import os
import csv
from preprocessing_mtcnn import get_landmarks
from extract_fetures_image_proc import our_mtcnn



def create_csv_file():
    csv_file = open('our_mtcnn.csv', 'w', newline='')
    obj = csv.writer(csv_file)
    arr = []
    arr.append('Emotion')
    arr.append('left_eye d')
    arr.append('left_eye a')
    arr.append('right_eye d')
    arr.append('right_eye a')
    arr.append('nose d')
    arr.append('nose a')
    arr.append('mouth_left d')
    arr.append('mouth_left a')
    arr.append('mouth_right d')
    arr.append('mouth_right a')

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
                arr_to_write = our_mtcnn(str(directory_name) + '/' + str(filename))
                print(arr_to_write)
                if arr_to_write !=  None:
                    print(str(directory_name) + '/' + str(filename))
                    arr_to_write.insert(0, emotion)
                    obj.writerow(arr_to_write)
    csv_file.close()


if __name__ == '__main__':
    create_csv_file()
