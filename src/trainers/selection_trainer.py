import torch
from torch.nn import CrossEntropyLoss

from trainers.base import BaseTrainer


class SelectionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def metrics_fn(self, y_hat, labels):
        return {"acc": torch.eq(y_hat.argmax(dim=1), labels).to(torch.float).mean()}

    def run_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(y_hat, batch["label"].to(self.device))
        return loss, self.metrics_fn(y_hat, batch["label"])

    def run_predict_batch(self, batch):
        pass
