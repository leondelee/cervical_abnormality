#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import os
import cv2
from tqdm import  tqdm

import torch as t
import numpy as np

from config import *


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dense_to_one_hot(label, num_of_classes):
    """
    Convert a dense representation of one vector to one-hot representation
    :param origin_tensor:
    :param num_of_classes:
    :return:
    """
    res = t.zeros(num_of_classes, ).long()
    res[label] = 1
    return res


def check_previous_models():
    """
    check whether there exist previous models, if true, then let user choose whether to use it.
    :return:
    """
    available_models = os.listdir(CHECK_POINT_PATH)
    available_models.sort(key=lambda x: get_time_stamp(x))
    while available_models:
        print('Do you want to keep and load previous models ?')
        key = input('Please type in k(keep) / d(delete): ')
        if key == 'k':
            model_name = CHECK_POINT_PATH + available_models[-1]
            return model_name
        elif key == 'd':
            for model in available_models:
                os.unlink(CHECK_POINT_PATH + model)
            return None
        else:
            print('Please type k or d !')


def mylog(file_name, log_content):
    """
    define a logger function
    :param file_name:  the name of the log file
    :param log_content:  the content to be logged
    :return:
    """
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    with open(LOG_PATH + file_name + '.log', 'a+') as file:
        file.write(log_content + '\n')
        file.close()


def get_time_stamp(str, time_format=TIME_FORMAT):
    """
    given a time string in a particular format, get the time stamp represented by this string
    :param str: given time string
    :param time_format: time format used
    :return:  time stamp
    """
    import time
    import datetime
    import re
    timestr = re.findall('>_(.*)\.', str)[0]
    return time.mktime(datetime.datetime.strptime(timestr, time_format).timetuple())


def evaluate(model, metric, eval_data):
    model.eval()
    print("Evaluating model...\n")
    log_content = ""
    res = 0
    for data in tqdm(eval_data):
        X, y = data
        X = X.float().to(DEVICE)
        y = y.long().to('cpu')
        out = model(X)
        out = t.argmax(out, dim=1).cpu()
        res += metric(out, y)
    res = res / len(eval_data)
    log_content += "Average {metric_name} is {metric_value}.\n".format(metric_name=metric.__name__, metric_value=res)
    model.train()
    return log_content


if __name__ == '__main__':
    import os
    av = os.listdir(CHECK_POINT_PATH)
    av.sort(key=lambda x: get_time_stamp(x))
    print(av)