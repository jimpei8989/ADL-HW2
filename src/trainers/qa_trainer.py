from collections import OrderedDict

from torch.nn import CrossEntropyLoss

from datasets.utils import pack
from metrics.accuracy import Accuracy
from metrics.f1 import F1Score
from trainers.base import BaseTrainer


class QATrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def create_metrics(self):
        return OrderedDict(
            [
                ("acc", Accuracy(convert_fn=lambda ps, pe, gs, ge: (zip(ps, pe), zip(gs, ge)))),
                ("f1", F1Score()),
                ("s_acc", Accuracy(convert_fn=lambda ps, pe, gs, ge: (ps, gs))),
                ("e_acc", Accuracy(convert_fn=lambda ps, pe, gs, ge: (pe, ge))),
            ]
        )

    def run_batch(self, batch):
        start_logits, end_logits = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(start_logits, batch["start_index"].to(self.device)) + self.criterion(
            end_logits, batch["end_index"].to(self.device)
        )
        return loss, self.update_metrics(
            start_logits.cpu().argmax(dim=1),
            end_logits.cpu().argmax(dim=1),
            batch["start_index"],
            batch["end_index"],
        )

    def run_predict_batch(self, batch):
        start_indices, end_indices = map(
            lambda t: t.argmax(dim=-1), self.model(batch["input_ids"].to(self.device))
        )
        return pack(
            batch | {"start_index": start_indices.tolist(), "end_index": end_indices.tolist()}
        )
