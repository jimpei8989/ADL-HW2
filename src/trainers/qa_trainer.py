import torch
from torch.nn import CrossEntropyLoss

from trainers.base import BaseTrainer


class QATrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def metrics_fn(self, start_logits, end_logits, start_index, end_index):
        return {
            "acc": torch.logical_and(
                torch.eq(start_logits.argmax(dim=1), start_index),
                torch.eq(end_logits.argmax(dim=1), end_index),
            )
            .to(torch.float)
            .mean()
        }

    def run_batch(self, batch):
        start_logits, end_logits = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(start_logits, batch["start_index"].to(self.device)) + self.criterion(
            end_logits, batch["end_index"].to(self.device)
        )
        return loss, self.metrics_fn(
            start_logits, end_logits, batch["start_index"], batch["end_index"]
        )

    def run_predict_batch(self, batch):
        pass
