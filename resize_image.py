from matplotlib.pyplot import imshow

import matplotlib.image as img
import numpy as npy

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return npy.array((x - npy.min(x)) / (npy.max(x) - npy.min(x)))


def image_resize(image_path, newX, newY):
    # provide the location of image for reading
    m = img.imread(image_path);
    print(m.shape)
    # determining the length of original image
    w, h = m.shape[:2];
    print(w)
    print(h)
    # xNew and yNew are new width and
    # height of image required

    xNew = newX;
    yNew = newY;

    # calculating the scaling factor
    # work for more than 2 pixel
    xScale = xNew / (w - 1);
    yScale = yNew / (h - 1);

    print(xScale)
    print(yScale)
    # using numpy taking a matrix of xNew
    # width and yNew height with
    # 4 attribute [alpha, B, G, B] values
    newImage = npy.zeros([xNew, yNew, 3]);

    for i in range(xNew - 1):
        for j in range(yNew - 1):
            newImage[i + 1, j + 1] = m[1 + int(i / xScale), 1 + int(j / yScale)]

        # Save the image after scaling
    img.imsave('scaled.png', normalize(newImage));
