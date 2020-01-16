import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from skimage import color

from scipy.misc import imread, imsave


# Notice : get divided center.
# # image = imread("/data1/LJH/Dot_To_Mask_instance_segmentation/A1/plant001_centers_hd.png")
#
# # read image through command line
# img = cv2.imread("/data1/LJH/Dot_To_Mask_instance_segmentation/middle_result/test.png")
#
# # convert the image to grayscale
# gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
# # convert the grayscale image to binary image
# ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
#
# # find contours in the binary image
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     # calculate moments for each contour
#     M = cv2.moments(c)
#
#     # calculate x,y coordinate of center
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     cv2.circle(img, (cX, cY), 2, (0, 0, 0), -1)
#     # cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#     # display the image
#     plt.imshow(img)
#     plt.show()
#
#     # # prevent zero.
#     # if M["m00"] != 0:
#     #     cX = int(M["m10"] / M["m00"])
#     #     cY = int(M["m01"] / M["m00"])
#     # else:
#     #     cX, cY = 0, 0

def get_centers(image_path):
    # using this code.
    # https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/

    img = cv2.imread(image_path)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

    centers = []
    coordinates = []

    # find contours in the binary image
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for index, c in enumerate(contours):
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # prevent zero.
            cX, cY = 0, 0

        coordinates.append([cX, cY])
        canvas = np.zeros(shape=[np.shape(img)[0], np.shape(img)[1]])

        cv2.circle(canvas, (cX, cY), 5, 255, -1)
        # cv2.imwrite(image_path.replace(".png", "") + "_center{}.png".format(index), canvas)

        centers.append(canvas / 255)

    return centers, coordinates


def get_areas_from_center(image_path, centers):
    color_label = imread(image_path)
    color_centers = []

    for index, center in enumerate(centers):
        selected_color = color_label[center[1], center[0], :]
        canvas = np.zeros(shape=np.shape(color_label))
        canvas[np.where(np.all(color_label == selected_color, axis=-1))] = 255

        # imsave(image_path.replace(".png", "") + "_center{}.png".format(index), canvas)

        canvas = color.rgb2gray(canvas)

        color_centers.append(canvas / 255)

    return color_centers

#
# img, index = get_centers("/data1/LJH/Dot_To_Mask_instance_segmentation/A1/plant001_centers_hd.png")
# color_centers = get_areas_from_center("/data1/LJH/Dot_To_Mask_instance_segmentation/A1/plant001_label.png", index)
# print("")
# Notice : input is divided center and segmented image.

# Notice : output is specific segmented area.
