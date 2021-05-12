import numpy as np

from metrics.base import BaseMetric


class F1Score(BaseMetric):
    @staticmethod
    def calculate(ps, pe, gs, ge):
        def clip(x, a, b):
            return a if x < a else b if x > b else x

        if gs >= ge:
            return np.nan
        elif ps > pe:
            return 0
        else:
            overlap = clip(pe, gs, ge) - gs if ps < gs else clip(ge, ps, pe) - ps
            presicion = overlap / (ge - gs)
            recall = overlap / (pe - ps)
            return 2 * (presicion * recall) / (presicion + recall)

    def __init__(self, convert_fn=None):
        self.epoch = []
        self.buffer = []
        self.convert_fn = convert_fn

    def initialize_epoch(self):
        self.buffer.clear()

    def _update( self, p_start, p_end, g_start, g_end):
        f1_scores = [self.calculate(*pgpg) for pgpg in zip(p_start, p_end, g_start, g_end)]
        self.buffer.extend(filter(lambda x: not np.isnan(x), f1_scores))
        return np.mean(self.buffer[-min(len(self.buffer), 16):])

    def finalize_epoch(self):
        epoch_average = np.mean(self.buffer)
        self.epoch.append(epoch_average)
        return epoch_average
