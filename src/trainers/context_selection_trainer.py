import torch
from torch.nn import BCEWithLogitsLoss

from trainers.base import BaseTrainer


class ContextSelectionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = BCEWithLogitsLoss()

    def metrics_fn(self, y_hat, labels):
        return {"acc": torch.eq(y_hat.argmax(dim=1), labels.argmax(dim=1)).to(torch.float).mean()}

    def run_batch(self, batch):
        y_hat = self.model(batch["sel_input_ids"].to(self.device))
        loss = self.criterion(y_hat, batch["sel_label"].to(self.device))
        return loss, self.metrics_fn(y_hat.cpu(), batch["sel_label"])

    def run_predict_batch(self, batch):
        pass
