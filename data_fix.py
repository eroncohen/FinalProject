import os
import csv
from matplotlib import pyplot as plt
from binary_image import crop_mouth_from_face


def get_pixels_from_image(image_path):
    #im = Image.open(imagePath, 'r')
    #pix_val = list(im.getdata())
    #pix_val_flat = [x for sets in pix_val for x in sets]

    image = plt.imread(image_path)
    #img = smp.toimage(image)  # Create a PIL image
    #img.show()
    img_str = ' '.join(map(str, image.flatten()))
    print(len(img_str.split(" ")))
    return img_str


if __name__ == "__main__":

    csvfile = open('mouth_only_pic.csv', 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(['emotion', 'pixels', 'Usage'])
    directoryNameList = ['Training', 'PublicTest', 'PrivateTest']

    directory = os.fsencode('C:/Users/Eron/PycharmProjects/Final_Project')
    counter = 0
    for name in directoryNameList:
        for x in range(7):
            if x == 3:
                cls = 1
            else:
                cls = 0
            directoryName = 'C:/Users/Eron/PycharmProjects/Final_Project/' + name + '/' + str(x)
            directory = os.fsencode(directoryName)

            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                temp_image = crop_mouth_from_face(str(directoryName) + '/' + str(filename))
                obj.writerow([cls, temp_image, name])
    csvfile.close()
