# Author: llw
import os
import sys

import cv2
import numpy as np


path = "../data/cervical_data_with_vgg/vinegar/vinegar"
test_path = os.path.join(path, "test")
train_path = os.path.join(path, "train")
val_path = os.path.join(path, "validation")


def get_image_dir(cat="train", type="2"):
    if cat == "train":
        return os.path.join(train_path, type)
    elif cat == "test":
        return os.path.join(test_path, type)
    else:
        return os.path.join(val_path, type)


def get_image_path(dir, image):
    return os.path.join(dir, image)


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_array(filename):
    return cv2.imread(filename)


if __name__ == '__main__':
    dir_path = get_image_dir("train", "2")
    imgs_path = os.listdir(dir_path)
    for img_path in imgs_path:
        img_path = get_image_path(dir_path, imgs_path[0])
        img = get_image_array(img_path)
        print(img.shape)