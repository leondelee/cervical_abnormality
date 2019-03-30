#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name : Trainer,py
File Description : Define the Trainer class for model training
Author : https://github.com/pytorch/pytorch/blob/master/torch/utils/trainer/trainer.py

"""
import heapq
import time
from tqdm import tqdm

import torch as t

from config import *
from tools.tools import *


class Trainer:
    """
    define a trainer class to train model
    """
    def __init__(self, model, criterion, optimizer, dataset, val_dataset, metric):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.max_epoch = MAX_EPOCH
        self.batch_size = BATCH_SIZE
        self.model = model
        self.learning_rate = LEARNING_RATE
        self.iterations = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        self.loss = 0
        self.metric = metric

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self):
        print('Start training...')
        for q in self.plugin_queues.values():
            heapq.heapify(q)
        current_time = time.strftime(TIME_FORMAT)
        for epoch in range(self.max_epoch):
            print('Epoch', epoch)
            self.step_one_epoch()
            self.call_plugins('epoch', epoch)
            if (epoch + 1) % UPDATE_FREQ == 0:
                if MODEl_SAVE:
                    self.model.save()
                log_content = 'Loss at epoch {epoch} is {loss}.\n'.format(
                    epoch=epoch,
                    loss=self.loss,
                )
                log_content += evaluate(self.model, self.metric, self.val_dataset)
                print(log_content)
                mylog(current_time, log_content)

    def step_one_epoch(self):
        self.loss = 0
        for iteration, data in enumerate(tqdm(self.dataset)):
            batch_input, batch_label = data
            batch_input = batch_input.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            self.call_plugins('batch', iteration, batch_input, batch_label)
            input_var = t.autograd.Variable(batch_input).float()
            label_var = t.autograd.Variable(batch_label).long()
            plugin_data = [None, None]

            def closure():
                output_var = self.model(input_var).float()
                self.loss = self.criterion(output_var, label_var)
                self.loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = output_var.data
                    plugin_data[1] = self.loss
                return plugin_data[1]
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', iteration, batch_input, batch_label, *plugin_data)
            self.call_plugins('update', iteration, self.model)
        self.iterations += iteration




