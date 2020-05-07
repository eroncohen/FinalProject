import matplotlib.image as img
import numpy as np


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def image_resize_to_square(image, new_width, new_height):
    m = image

    # get dimensions of image
    dimensions = image.shape
    # height, width, number of channels in image
    height = image.shape[0]
    width = image.shape[1]

    width_to_crop = (width - new_width)/2
    height_to_crop = (height - new_height)/2

    new_image = np.zeros([new_width, new_height])
    # crop width
    new_image_x = 0
    new_image_y = 0
    if width_to_crop >= 0 and height_to_crop >= 0:
        for old_image_x in range(int(height_to_crop), int(height - height_to_crop - 1)):
            for old_image_y in range(int(width_to_crop), int(width - width_to_crop-1)):
                new_image[new_image_x, new_image_y] = m[int(old_image_x), int(old_image_y)]
                new_image_y += 1
            new_image_y = 0
            new_image_x += 1
    return new_image


def image_resize(image, new_x, new_y):
    # provide the location of image for reading
    m = image
    # determining the length of original image
    w, h = m.shape[:2]

    # work for more than 2 pixel
    x_scale = new_x / (w - 1)
    y_scale = new_y / (h - 1)

    # using numpy taking a matrix of xNew
    # width and yNew height with
    # 4 attribute [alpha, B, G, B] values
    new_image = np.zeros([new_x, new_y])

    for i in range(new_x - 1):
        for j in range(new_y - 1):
            new_image[i + 1, j + 1] = m[1 + int(i / x_scale), 1 + int(j / y_scale)]
    return new_image


def crop_mouth_from_face(image, is_cnn):
    mouth_vector = []
    # reading image in variable m
    # m = img.imread(img_path)
    # determining dimesion of image width(w) height(h)
    w, h = image.shape[:2]

    # required image size after cropping
    x_new = int(w * 1 / 3)
    y_new = int(h * 1 / 3)
    new_image = np.zeros([x_new, y_new])

    for i in range(1, x_new):
        for j in range(1, y_new):
            # x cropping the last third of the picture, y cropping the 2nd third
            new_image[i, j] = image[x_new * 2 + i, y_new + j]
            mouth_vector.append(image[x_new * 2 + i, y_new + j])
    return mouth_vector if not is_cnn else new_image


def crop_mouth_without_cv(image):
    # reading image in variable m
    # m = img.imread("small_image.jpg")
    m = image
    # determining dimension of image width(w) height(h)
    w, h = m.shape[:2]

    # required image size after cropping
    x_new = int(w * 1 / 3)
    y_new = int(h * 1 / 3)

    x_new_eyes = int(w * 1 / 4)
    y_new_eyes = int(h * 1 / 4)

    new_image_mouth = np.zeros([x_new, y_new])
    new_image_eyes = np.zeros([x_new_eyes, y_new_eyes*2, 3])

    for i in range(1, x_new):
        for j in range(1, y_new):
            new_image_mouth[i, j] = m[x_new*2 + i, y_new+j]

    for i in range(1, x_new_eyes):
        for j in range(1, y_new_eyes*2):
            new_image_eyes[i, j] = m[x_new_eyes + i, y_new_eyes + j]

    return new_image_mouth


def crop_face(gray_image, x, y, w, h):
    r = max(w, h) / 2
    center_x = x + w / 2
    center_y = y + h / 2
    nx = int(center_x - r)
    ny = int(center_y - r)
    nr = int(r * 2)
    return gray_image[ny:ny + nr, nx:nx + nr]
