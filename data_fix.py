import os
import csv
from matplotlib import pyplot as plt

def getPixelFromImage(imagePath):
    #im = Image.open(imagePath, 'r')
    #pix_val = list(im.getdata())
    #pix_val_flat = [x for sets in pix_val for x in sets]

    image = plt.imread(imagePath)
    #img = smp.toimage(image)  # Create a PIL image
    #img.show()
    img_str = ' '.join(map(str,image.flatten()))
    print(len(img_str.split(" ")))
    return img_str



if __name__== "__main__":

    csvfile = open('newFer2013.csv', 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(['emotion','pixels','Usage'])
    directoryNameList = ['Training', 'PublicTest', 'PrivateTest']

    directory = os.fsencode('C:/Users/Eron/PycharmProjects/Final_Project')
    counter = 0
    for name in directoryNameList:
        for x in range(7):
            directoryName = 'C:/Users/Eron/PycharmProjects/Final_Project/' + name + '/' + str(x)
            directory = os.fsencode(directoryName)

            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                temp_image = getPixelFromImage(str(directoryName) + '/' + str(filename))
                obj.writerow([x, temp_image, name])
    csvfile.close()
