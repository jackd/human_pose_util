import numpy as np
import cv2

# Bounding box of image
#
# The bounding box of the image calculated by the given 2D human pose


class BoundingBox(object):
    # constructor
    # @param image The base image
    # @param x_2d The 2d pose of human
    # @param image_size The size of output image, the default value is (256x256)
    # @param cropping_size The size of cropping for DNN training, the default value is (227x227)
    # @param margin The cropping margin, default value is 10
    def __init__(self, image, x_2d, image_size=256, cropping_size=227, margin=10):
        if 2 * (cropping_size - margin) - image_size <= 0:
            raise ValueError(
                "Bad image and cropping size, 2 x (cropping_size({0}) - margin({1})) - image_size({2}) should be > 0".format(cropping_size, margin, image_size))
        # calculate bounding box
        x_2d_list = x_2d.get()[1]
        x = [float(v[0]) for v in x_2d_list]
        y = [float(v[1]) for v in x_2d_list]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        # check the existence of solution
        exist = True
        c = float(cropping_size - margin)
        s = float(image_size)
        h, w, _ = image.shape
        p_min, p_max = np.array([x_min, y_min]), np.array([x_max, y_max])
        if ((-(s - c) * p_max + c * p_min) / (2 * c - s) < 0).any() or ((c * p_min - (s - c) * p_min) / (2 * c - s) > np.array([w, h])).any():
            exist = False
        l = s / (2 * c - s)
        if (x_max - x_min) > (y_max - y_min):
            if h < l * (x_max - x_min) or (s - c) * (x_max - x_min) > (2 * c - s) * min(y_min, h - y_max):
                exist = False
        else:
            if w < l * (y_max - y_min) or (s - c) * (y_max - y_min) > (2 * c - s) * min(x_min, w - x_max):
                exist = False
        if not exist:
            raise RuntimeError(
                "The assumption is broken that the person in the screen is not big so much and all marker is seen after cropping.")
        # extend bounding box considering cropping size
        k = 1.0 * max(x_max - x_min, y_max - y_min) / (2 * c - s)
        if (x_max - x_min) > (y_max - y_min):
            u_min = x_min - k * (s - c)
            u_max = x_max + k * (s - c)
            w_2 = (u_max - u_min) / 2
            y_c = (y_max + y_min) / 2
            if y_c + w_2 < h:
                y_c = h - w_2
            elif y_c < w_2:
                y_c = w_2
            v_min = y_c - w_2
            v_max = y_c + w_2
        else:
            v_min = y_min - k * (s - c)
            v_max = y_max + k * (s - c)
            h_2 = (v_max - v_min) / 2
            x_c = (x_max + x_min) / 2
            if w - h_2 < x_c:
                x_c = w - h_2
            elif x_c < h_2:
                x_c = h_2
            u_min = x_c - h_2
            u_max = x_c + h_2
        # discretization
        u_min, u_max, v_min, v_max = int(np.floor(u_min)), int(
            np.ceil(u_max)), int(np.floor(v_min)), int(np.ceil(v_max))
        h = max(u_max - u_min, v_max - v_min)
        k = 1. * h / image_size
        cropping_image = image[v_min:v_min + h, u_min:u_min + h]
        # the image cropped by bounding box
        self.image = cv2.resize(cropping_image, (image_size, image_size))
        # bounding box image coordinate origin
        self.u_0 = np.matrix([u_min, v_min, 0.]).T
        # scaling factor
        self.s = 1. / k
