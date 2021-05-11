from collections import OrderedDict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from trainers.base import BaseTrainer


class QATrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def f1_score(self, ps, pe, gs, ge):
        def clip(x, a, b):
            return a if x < a else b if x > b else x

        if gs == ge:
            return np.nan
        elif ps > pe:
            return 0
        else:
            overlap = clip(pe, gs, ge) - gs if ps < gs else clip(ge, ps, pe) - ps
            presicion = overlap / (ge - gs)
            recall = overlap / (pe - ps)
            return 2 * (presicion * recall) / (presicion + recall)

    def metrics_fn(self, start_logits, end_logits, start_index, end_index):
        pred_start = start_logits.argmax(dim=1)
        pred_end = end_logits.argmax(dim=1)

        start_correct = torch.eq(pred_start, start_index)
        end_correct = torch.eq(pred_end, end_index)

        f1_score = np.mean(
            [
                self.f1_score(ps, pe, gs, ge)
                for ps, pe, gs, ge in zip(pred_start, pred_end, start_index, end_index)
            ]
        )

        return OrderedDict(
            {
                "acc": torch.logical_and(start_correct, end_correct).to(torch.float).mean(),
                "f1": f1_score,
                "start_acc": start_correct.to(torch.float).mean(),
                "end_acc": end_correct.to(torch.float).mean(),
            }
        )

    def run_batch(self, batch):
        start_logits, end_logits = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(start_logits, batch["start_index"].to(self.device)) + self.criterion(
            end_logits, batch["end_index"].to(self.device)
        )
        return loss, self.metrics_fn(
            start_logits.cpu(), end_logits.cpu(), batch["start_index"], batch["end_index"]
        )

    def run_predict_batch(self, batch):
        pass
