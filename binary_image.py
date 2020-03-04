import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as img


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def crop_mouth_from_face(img_path):
    arr = []
    # reading image in variable m
    m = img.imread(img_path)
    # determining dimesion of image width(w) height(h)
    w, h = m.shape[:2]

    # required image size after cropping
    xNew = int(w * 1 / 3)
    yNew = int(h * 1 / 3)
    newImage = np.zeros([xNew, yNew, 3])

    # print width height of original image
    print(xNew)
    print(yNew)

    for i in range(1, xNew):
        for j in range(1, yNew):
            newImage[i, j] = m[xNew * 2 + i, yNew + j]
            arr.append(m[xNew * 2 + i, yNew + j])
    print(arr)
    return arr
        # save image
    #img.imsave('cropped.png', normalize(newImage))


if __name__ == "__main__":
    crop_mouth_from_face()
    image = cv2.imread(
        'cropped.png')  # C:/Users/Eron/Desktop/PublicTest/5/PublicTest_57538899.jpg
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        print(i)
        x, y = i.ravel()
        cv2.circle(image, (x, y), 1, 255, -1)

    plt.imshow(image), plt.show()
