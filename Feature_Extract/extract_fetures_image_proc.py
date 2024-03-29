import numpy as np
import cv2
import math
MARGIN_SIZE = 6


def mean_of_image(image):
    pixel_sum = 0
    counter = 0
    for i in range(MARGIN_SIZE, len(image)-MARGIN_SIZE):
        for j in range(MARGIN_SIZE, len(image[0])-MARGIN_SIZE):
            pixel_sum += image[i, j]
            counter += 1
    return pixel_sum/counter


def find_eyes(image, mean_pixels):
    left_part = []
    right_part = []
    half_mean = mean_pixels / 4 * 3
    for i in range(10, 20):
        for j in range(12, 36):
            if image[i, j] < half_mean:
                if check_if_eye(image, i, j, half_mean):
                    if j < 24:
                        left_part.append((j, i))
                    else:
                        right_part.append((j, i))
    if len(left_part) == 0 or len(right_part) == 0:
        return None, None
    x_sum = 0
    y_sum = 0
    for x in left_part:
        x_sum += x[0]
        y_sum += x[1]
    x_left_eye_mean = x_sum / len(left_part)
    y_left_eye_mean = y_sum / len(left_part)
    x_sum = 0
    y_sum = 0
    for x in right_part:
        x_sum += x[0]
        y_sum += x[1]
    x_right_eye_mean = x_sum / len(right_part)
    y_right_eye_mean = y_sum / len(right_part)
    return (round(x_left_eye_mean), round(y_left_eye_mean)), (round(x_right_eye_mean), round(y_right_eye_mean))


def check_if_eye(image, x, y, mean):
    pixels_sum = 0
    #if (image[x-1, y] < mean and image[x-2, y] < mean and image[x+1, y] < mean and image[x+2, y] < mean and
            #image[x, y-1] < mean and image[x, y+1] < mean):
    #pixels_mean = (image[x - 1, y] + image[x + 1, y] + image[x, y - 1] + image[x, y - 2] + image[x, y + 2] + image[x, y + 1] + image[x, y])/7

    pixels_sum += image[x, y]
    pixels_sum += image[x - 1, y]
    pixels_sum += image[x + 1, y]
    pixels_sum += image[x, y - 1]
    pixels_sum += image[x, y - 2]
    pixels_sum += image[x, y + 1]
    pixels_sum += image[x, y + 2]
    pixels_mean = pixels_sum/7

    #pixels_sum = (int(image[x - 1, y] + image[x + 1, y] + image[x, y - 1] + image[x, y - 2] + image[x, y + 2] + image[x, y + 1] + image[x, y]))/7
    #pixels_arr = np.array([image[x - 1, y], image[x + 1, y], image[x, y - 1], image[x, y - 2], image[x, y + 2], image[x, y + 1], image[x, y]])
    #pixels_mean = np.mean(pixels_arr)
    if pixels_mean < mean:
        return True
    else:
        return False


def find_mouth(image, mean_pixels):
    left_part = []
    right_part = []
    half_mean = mean_pixels / 4 * 3
    for i in range(32, 42):
        for j in range(12, 20):
            if image[i, j] < half_mean:
                if check_if_mouth(image, i, j, half_mean, is_left=1):
                    left_part.append((j, i))
        for k in range(28, 36):
            if image[i, k] < half_mean:
                if check_if_mouth(image, i, k, half_mean, is_left=0):
                    right_part.append((k, i))
    if len(left_part) == 0 or len(right_part) == 0:
        return None, None
    left_mouth = left_part[0]
    for x in left_part:
        if x[0] < left_mouth[0]:
            left_mouth = x
    right_mouth = right_part[0]
    for x in right_part:
        if x[0] > right_mouth[0]:
            right_mouth = x
    return left_mouth, right_mouth


def check_if_mouth(image, x, y, mean, is_left):
    if is_left == 1:
        if all([image[x, y+1] < mean, image[x, y+2] < mean, image[x, y-1] > mean, image[x, y-2] > mean]):
            return True
        else:
            return False
    else:
        if all([(image[x, y-1] < mean), (image[x, y-2] < mean), (image[x, y+1] > mean), (image[x, y+2] > mean)]):
            return True
        else:
            return False


def find_nose(image, mean_pixels):
    nose_part = []
    for i in range(20, 30):
        for j in range(20, 28):
            if image[i, j] < mean_pixels:
                nose_part.append((j, i))
    if len(nose_part) == 0:
        return None
    x_nose_sum = 0
    y_nose_sum = 0

    for point in nose_part:
        x_nose_sum = x_nose_sum + point[0]
        y_nose_sum = y_nose_sum + point[1]
    return round(x_nose_sum/len(nose_part)), round(y_nose_sum/len(nose_part))


def ye_algorithm_detect_five_points(image_path, image):
    if image_path is not None:
        img_to_detect = cv2.imread(image_path)
    else:
        img_to_detect = image
    mean_pixels = mean_of_image(img_to_detect)
    left_eye, right_eye = find_eyes(img_to_detect, mean_pixels)
    left_m, right_m = find_mouth(img_to_detect, mean_pixels)
    nose = find_nose(img_to_detect, mean_pixels)
    points_list = []
    if left_eye is not None and left_m is not None and nose is not None:
        points_list.append(left_eye)
        points_list.append(right_eye)
        points_list.append(nose)
        points_list.append(left_m)
        points_list.append(right_m)
        return points_list
    else:
        return None


if __name__ == "__main__":
    # imag = cv2.imread("C:/Users/Eron/PycharmProjects/Final_Project/PrivateTest/3/PrivateTest_325008.jpg")
    imag = cv2.imread("C:/Users/Eron/Desktop/Black_photo.jpg")
    mean_pixel = mean_of_image(imag)
    print(mean_pixel)
    '''
    left_eye, right_eye = find_eyes(imag, mean_pixels)
    left_m, right_m = find_mouth(imag, mean_pixels)
    nose = find_nose(imag, mean_pixels)
    print(left_eye)
    print(right_eye)
    print(left_m)
    print(right_m)
    print(nose)
    if left_eye is not None and left_m is not None and nose is not None:
        cv2.circle(imag, left_eye, 1, (255, 0, 0), 1)
        cv2.circle(imag, right_eye, 1, (255, 0, 0), 1)
        cv2.circle(imag, left_m, 1, (255, 0, 0), 1)
        cv2.circle(imag, right_m, 1, (255, 0, 0), 1)
        cv2.circle(imag, nose, 1, (255, 0, 0), 1)
        cv2.imshow("img", imag)
        cv2.waitKey(0)
    '''

