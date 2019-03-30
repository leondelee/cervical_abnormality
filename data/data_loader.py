import sys
sys.path.append("../../cervical_abnormality")
import os

import cv2
from PIL import Image
from torch.utils import data as DT
from torchvision import transforms as TM

# TODO :from tools.image_preprocess import transform
from config import *
from tools.tools import show_image


def get_transformer():
    if RESIZE:
        return TM.Compose(
            [
                TM.Resize(HEIGHT),
                TM.ToTensor(),
                TM.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ]
        )
    else:
        return TM.Compose(
            [
                TM.ToTensor(),
                TM.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ]
        )


my_transform = get_transformer()


def load_data(dataset, shuffle=True, drop_last=False):
    if len(dataset) > 1:
        dataset = DT.ConcatDataset(dataset)
    data_loader = DT.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=LOAD_DATA_WORKERS,
        drop_last=drop_last
    )
    return data_loader


class DataSet(DT.Dataset):

    def __init__(self, data_type="train", label=0, annotation_type=VINEGAR, transform=my_transform):
        self.data_type = data_type
        self.label = label
        self.annotation_type = annotation_type
        self.data_path = os.path.join(DATA_PATH, data_type)
        if annotation_type == VINEGAR:
            self.data_path = os.path.join(self.data_path, "vinegar")
        elif annotation_type == IODINE:
            self.data_path = os.path.join(self.data_path, "iodine")
        self.data_path = os.path.join(self.data_path, LEVEL_FOLDER[label])
        self.images_path = [os.path.join(self.data_path, image_path) for image_path in os.listdir(self.data_path)]

        if not transform:
            # TODO self.transform = 12345
            pass
        else:
            self.transform = transform

    def __getitem__(self, item):
        this_image = self.images_path[item]
        this_image_data = Image.open(this_image)
        this_image_data = self.transform(this_image_data)
        # this_image_data = this_image_data.reshape(NUM_CHANNELS, HEIGHT, WIDTH)
        return this_image_data, self.label

    def __len__(self):
        return len(self.images_path)

    def size(self):
        return self.__len__()

    def show(self, items):
        if type(items) == int:
            show_image(self.__getitem__(items))
        else:
            for item in items:
                show_image(self.__getitem__(item))


if __name__ == '__main__':
    ds = DataSet(data_type="train", label=0)
    print(ds.size())
    ds = load_data([ds, ds])
    for idx, i in enumerate(ds):
        print(i[0].shape)