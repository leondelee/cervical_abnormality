#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

File Name : main.py
File Description : This is the file where we do the training, testing, validating job
Author : Liangwei Li

"""
import torch as t
from sklearn.metrics import accuracy_score, f1_score

from tools.trainer import Trainer
from model.ResNet34 import ResNet34
from data.data_loader import *
from config import *
from tools.tools import check_previous_models, evaluate


def train(trainer):
    trainer.run()


if __name__ == '__main__':
    # define model
    # model = ResNet34(in_channels=NUM_CHANNELS, out_classes=NUM_CLASSES).to(DEVICE)
    from torchvision.models import resnet18
    model = resnet18(pretrained=True).to(DEVICE)
    if MODEl_SAVE:
        model_flag = check_previous_models()                       # check if there exist previous models
        if model_flag != None:
            model.load(model_flag)

    # load_data
    type_name = ["vinegar"]
    train_dic = val_dic = test_dic = {}
    for idx, name in enumerate(type_name):
        train_dic[name] = [DataSet(data_type="train", label=k, annotation_type=idx) for k in range(NUM_CLASSES)]
        val_dic[name] = [DataSet(data_type="validation", label=k, annotation_type=idx) for k in range(NUM_CLASSES)]
        test_dic[name] = [DataSet(data_type="test", label=k, annotation_type=idx) for k in range(NUM_CLASSES)]

    train_data = load_data([d for d in [name_data for name_data in train_dic[name] for name in type_name]])
    val_data = load_data([d for d in [name_data for name_data in val_dic[name] for name in type_name]])
    test_data = load_data([d for d in [name_data for name_data in test_dic[name] for name in type_name]])

    # define training details
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset=train_data,
        val_dataset=val_data,
        metric=f1_score
    )

    train(trainer)
    # evaluate(model=model, metric=accuracy_score, eval_data=val_data)
    # evaluate(model, test_data)
