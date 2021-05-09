import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils.io import json_load


class ChineseQADatasetForContextSelection(Dataset):
    @classmethod
    def from_json(cls, context_json: Path, data_json: Path, **kwargs):
        contexts = json_load(context_json)
        data = json_load(data_json)
        return cls(contexts, data, **kwargs)

    def __init__(
        self,
        contexts: Path,
        data: Path,
        num_classes: int = 7,
        tokenizer: Optional[BertTokenizer] = None,
        test: bool = False,
        use_selection: bool = True,
        use_span: bool = True,
    ):
        super().__init__()
        self.contexts = contexts
        self.data = data
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.test = test
        self.use_selection = use_selection
        self.use_span = use_span

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Parameters:
            index: int

        Returns:
            input_ids: a tensor of shape (N, L)
            label: an integer
        """
        question = self.data[index].get("question")
        relevant = self.data[index].get("relevant")
        nonrelevant = [p for p in self.data[index].get("paragraphs") if p != relevant]

        chosen = [relevant] + random.sample(
            nonrelevant, k=min(self.num_classes - 1, len(nonrelevant))
        )
        random.shuffle(chosen)

        question_tokens = self.tokenizer.tokenize(question)
        target_length = 512 - len(question_tokens) - 3
        paragraph_tokens = [self.tokenizer.tokenize(self.contexts[pid]) for pid in chosen]

        input_ids = [
            self.tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + question_tokens + ["[SEP]"] + para_tokens[:target_length] + ["[SEP]"]
            )
            for para_tokens in paragraph_tokens
        ] + [[self.tokenizer.cls_token_id]] * (self.num_classes - len(chosen))

        input_tensor = torch.stack(
            [
                torch.as_tensor(ids + [self.tokenizer.pad_token_id] * (512 - len(ids)))
                for ids in input_ids
            ]
        )

        return {
            "input_ids": input_tensor,
            "label": torch.as_tensor(
                [int(i == chosen.index(relevant)) for i in range(self.num_classes)],
                dtype=torch.float
            ),
        }
