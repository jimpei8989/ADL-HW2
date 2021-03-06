import torch
from datasets.base import BaseDataset


class ContextDataset(BaseDataset):
    TO_BE_PADDED = ["input_ids"]

    def __init__(self, *args, include_nonrelevant=1, **kwargs):
        super().__init__(*args, include_nonrelevant=include_nonrelevant, **kwargs)

    def __getitem__(self, index: int):
        """
        Parameters:
            index: int

        Returns:
            input_ids: a tensor of shape (N, L)
            label: an integer
        """
        d = self.data[index]
        question_tokens = d["question_tokens"]
        paragraph_tokens = d["paragraph_tokens"]

        input_ids = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token]
            + question_tokens
            + [self.tokenizer.sep_token]
            + paragraph_tokens
            + [self.tokenizer.sep_token]
        )

        ret = {"input_ids": torch.as_tensor(input_ids, dtype=torch.long)}
        if self.test:
            ret |= d
        else:
            ret |= {"label": float(d["has_answer"])}
        return ret
