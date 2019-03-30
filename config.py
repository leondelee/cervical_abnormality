# Author: llw
import os
import sys

import torch as t

# path information
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT_PATH)
DATA_PATH = os.path.join(ROOT_PATH, "data/")
MODEL_PATH = os.path.join(ROOT_PATH, "model/")
TOOLS_PATH = os.path.join(ROOT_PATH, "tools/")
CHECK_POINT_PATH = os.path.join(ROOT_PATH, "checkpoints/")
LOG_PATH = os.path.join(ROOT_PATH, "log/")

# device
DEVICE = 'cuda'

# image information
VINEGAR = 0
IODINE = 1
NUM_CHANNELS = 3
NUM_CLASSES = 2
RESIZE = False
HEIGHT = 224
WIDTH = 224
# disease level
"""
0 = 2
1 = 3
2 = cancer
"""
LEVELS = [l for l in range(NUM_CLASSES)]
LEVEL_FOLDER = os.listdir(os.path.join(DATA_PATH, "train/vinegar"))

# Training Details
BATCH_SIZE = 4
LOAD_DATA_WORKERS = 4
TIME_FORMAT = "%m_%d_%H:%M:%S"
MAX_EPOCH = 200
USE_GPU = True
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.99
UPDATE_FREQ = MAX_EPOCH / 10


if __name__ == '__main__':
    print(os.path.dirname(os.path.realpath(__file__)))

