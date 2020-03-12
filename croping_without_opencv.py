# using matplotlib and numpy
import matplotlib.image as img
import numpy as np


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def crop_mouth(image):
    # reading image in variable m
    #m = img.imread("small_image.jpg")
    m = image
    # determining dimesion of image width(w) height(h)
    w, h = m.shape[:2]

    # required image size after cropping
    xNew = int(w * 1 / 3)
    yNew = int(h * 1 / 3)

    xNewEyes = int(w * 1 / 4)
    yNewEyes = int(h * 1 / 4)

    newImageMouth = np.zeros([xNew, yNew])
    newImageEyes = np.zeros([xNewEyes, yNewEyes*2, 3])

    # print width height of original image
    #print(xNew)
    #print(yNew)

    for i in range(1, xNew):
        for j in range(1, yNew):
            newImageMouth[i, j] = m[xNew*2 + i, yNew+j]

    for i in range(1, xNewEyes):
        for j in range(1, yNewEyes*2):
            newImageEyes[i, j] = m[xNewEyes + i, yNewEyes + j]


    #print(normalize(newImageMouth))
        # save image
    #img.imsave('croppedMouth.png', normalize(newImageMouth))
    #img.imsave('croppedEyes.png', normalize(newImageEyes))
    return newImageMouth