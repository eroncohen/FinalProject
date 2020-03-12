from matplotlib.pyplot import imshow
import cv2
import matplotlib.image as img
import numpy as npy

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return npy.array((x - npy.min(x)) / (npy.max(x) - npy.min(x)))


def image_resize_to_square(image, newWidth, newHeight):
    m = image

    # get dimensions of image
    dimensions = image.shape
    # height, width, number of channels in image
    height = image.shape[0]
    width = image.shape[1]

    width_to_crop = (width - newWidth)/2
    height_to_crop = (height - newHeight)/2

    newImage = npy.zeros([newWidth, newHeight])
    # crop widht
    t = 0
    k = 0
    if width_to_crop >= 0 and height_to_crop>=0:
        for j in range(int(height_to_crop), int(height - height_to_crop - 1)):
            for i in range(int(width_to_crop) ,int(width - width_to_crop-1)):
                newImage[t, k] = m[int(j), int(i)]
                k += 1
            k = 0
            t += 1
    #img.imsave('after_squared.png', newImage)
    return newImage


def image_resize(image, newX, newY):
    # provide the location of image for reading
    m = image
    # determining the length of original image
    w, h = m.shape[:2]

    # xNew and yNew are new width and
    # height of image required

    xNew = newX
    yNew = newY

    # calculating the scaling factor
    # work for more than 2 pixel
    xScale = xNew / (w - 1)
    yScale = yNew / (h - 1)

    # using numpy taking a matrix of xNew
    # width and yNew height with
    # 4 attribute [alpha, B, G, B] values
    newImage = npy.zeros([xNew, yNew])

    for i in range(xNew - 1):
        for j in range(yNew - 1):
            newImage[i + 1, j + 1] = m[1 + int(i / xScale), 1 + int(j / yScale)]

        # Save the image after scaling
    #img.imsave('scaled.png', normalize(newImage))
    return newImage


if __name__ == "__main__":
    window_name = "Live Video Feed"
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_counter = 0  # to sample every 5 frames
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    while ret:
        if frame_counter % 20 == 0:  # Sends every 5 frame for detection
            ret, frame = cap.read()
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img.imsave('before_squared.png', frame)
            new_image = image_resize_to_square(gray_image, 480, 480)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyWindow(window_name)
    cap.release()