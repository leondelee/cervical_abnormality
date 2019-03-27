import sys
sys.path.append("../../cervical_abnormality")
import os

import cv2
from torch.utils import data as DT

# TODO :from tools.image_preprocess import transform
from config import *
from tools.image_preprocess import show_image


class DataSet(DT.Dataset):

    def __init__(self, data_type="train", label=LEVEL0, annotation_type=VINEGAR, transform=None):
        self.data_type = data_type
        self.annotation_type = annotation_type
        self.data_path = os.path.join("", data_type)
        self.data_path = os.path.join(self.data_path, str(label))
        if annotation_type == VINEGAR:
            self.data_path = os.path.join(self.data_path, "vinegar")
        elif annotation_type == IODINE:
            self.data_path = os.path.join(self.data_path, "iodine")
        self.images_path = os.listdir(self.data_path)

        if not transform:
            # TODO self.transform = 12345
            pass
        else:
            self.transform = transform

    def __getitem__(self, item):
        this_image = os.path.join(self.data_path, self.images_path[item])
        this_image_data = cv2.imread(this_image)
        # this_image_data = self.transform(this_image_data)
        return this_image_data

    def __len__(self):
        return len(self.images_path)

    def show(self, items):
        if type(items) == int:
            show_image(self.__getitem__(items))
        else:
            for item in items:
                show_image(self.__getitem__(item))


if __name__ == '__main__':
    ds = DataSet(data_type="train", label=0)
    ds.show([1, 2, 3])
