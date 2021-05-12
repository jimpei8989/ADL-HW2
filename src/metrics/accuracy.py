import numpy as np
import torch

from metrics.base import BaseMetric


class Accuracy(BaseMetric):
    @staticmethod
    def calculate(prediction, groundtruth):
        return torch.eq(prediction, groundtruth).to(torch.float).tolist()

    def __init__(self, convert_fn=None):
        self.epoch = []
        self.buffer = []
        self.convert_fn = convert_fn

    def initialize_epoch(self):
        self.buffer.clear()

    def _update(self, prediction, groundtruth):
        acc = self.calculate(prediction, groundtruth)
        print(prediction, groundtruth, acc)
        self.buffer.extend(acc)
        return np.mean(self.buffer[-min(len(self.buffer), 16):])

    def finalize_epoch(self):
        epoch_average = np.mean(self.buffer)
        self.epoch.append(epoch_average)
        return epoch_average
