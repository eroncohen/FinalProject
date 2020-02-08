import os
import csv
from preprocessing_dlib import get_landmarks


def create_csv_file():
    csv_file = open('landsmarkMouthAndEyeBrows.csv', 'w', newline='')
    obj = csv.writer(csv_file)
    arr = []
    arr.append('Emotion')
    for i in range(18, 28):
        arr.append(str(i) + "d")
        arr.append(str(i) + "a")
    for i in range(49, 61):
        arr.append(str(i) + "d")
        arr.append(str(i) + "a")
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
                arr_to_write = get_landmarks(str(directory_name) + '/' + str(filename))
                if arr_to_write != 0:
                    print(str(directory_name) + '/' + str(filename))
                    arr_to_write.insert(0, emotion)
                    obj.writerow(arr_to_write)
    csv_file.close()


if __name__ == '__main__':
    create_csv_file()