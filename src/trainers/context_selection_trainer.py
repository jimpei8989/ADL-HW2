from collections import OrderedDict

import torch
from torch.nn import BCEWithLogitsLoss

from trainers.base import BaseTrainer
from metrics.accuracy import Accuracy


class ContextSelectionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = BCEWithLogitsLoss()

    def create_metrics(self):
        return OrderedDict([("acc", Accuracy(convert_fn=lambda p, q: (torch.sigmoid(p).round(), q)))])

    def run_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(y_hat, batch["label"].to(self.device))
        return loss, self.update_metrics(y_hat.cpu(), batch["label"])

    def run_predict_batch(self, batch):
        pass
