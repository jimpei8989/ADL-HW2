import random
from itertools import chain
from pathlib import Path
from typing import List, Optional

from torch.utils.data import Dataset
from transformers import BertTokenizer

from datasets.utils import split_context_and_tokenize
from utils.io import json_dump, json_load
from utils.logger import logger
from utils.tqdmm import tqdmm


class BaseDataset(Dataset):
    @classmethod
    def from_json(cls, context_json: Path, data_json: Path, **kwargs):
        contexts = json_load(context_json)
        data = json_load(data_json)
        return cls(contexts, data, **kwargs)

    @classmethod
    def from_data(cls, data, **kwargs):
        dataset = cls(None, None, skip_preprocess=True, **kwargs)
        dataset.data = data
        return dataset

    def __init__(
        self,
        contexts: List[str],
        data: List[dict],
        tokenizer: Optional[BertTokenizer] = None,
        test: bool = False,
        include_nonrelevant=0,
        split_name: str = "no_name",
        cache_dir: Optional[Path] = None,
        skip_preprocess: Optional[bool] = False,
    ):
        super().__init__()
        self._contexts = contexts
        self._raw_data = data
        self.tokenizer = tokenizer
        self.test = test
        self.split_name = split_name

        if skip_preprocess:
            return

        cache_path = (
            (cache_dir / f"_{split_name}_preprocessed_{include_nonrelevant}.json")
            if cache_dir and split_name
            else None
        )

        if cache_path and cache_path.is_file():
            logger.info(f"Loading cached preprocessed dataset from {cache_path}...")
            self.data = json_load(cache_path)
        else:
            self.data = self.preprocess_dataset(
                self.tokenizer,
                contexts,
                data,
                include_nonrelevant=include_nonrelevant,
                test=self.test,
            )
            if cache_path:
                logger.info(f"Saving cached preprocessed dataset to {cache_path}...")
                json_dump(self.data, cache_path)

    @staticmethod
    def preprocess_dataset(tokenizer, contexts, data, include_nonrelevant=0, test=False):
        def extract(d):
            if test:
                ret = [
                    split_context_and_tokenize(
                        d["question"],
                        contexts[p],
                        tokenizer,
                        add_to_dict={"id": d["id"], "paragraph": p},
                    )
                    for p in d["paragraphs"]
                ]
            else:
                ret = [
                    split_context_and_tokenize(
                        d["question"],
                        contexts[d["relevant"]],
                        tokenizer,
                        d["answers"][0]["text"],
                        d["answers"][0]["start"],
                        d["answers"][0]["start"] + len(d["answers"][0]["text"]),
                        add_to_dict={"id": d["id"]},
                    )
                ]

                nonrelevant = random.sample(
                    [p for p in d["paragraphs"] if p != d["relevant"]],
                    k=min(len(d["paragraphs"]) - 1, include_nonrelevant),
                )
                ret.extend(
                    [
                        split_context_and_tokenize(
                            d["question"],
                            contexts[nonrel],
                            tokenizer,
                            add_to_dict={"id": d["id"]},
                        )
                        for nonrel in nonrelevant
                    ]
                )
            return chain.from_iterable(ret)

        preprocessed = list(
            chain.from_iterable(map(extract, tqdmm(data, desc="Preprocessing dataset")))
        )

        # Sanity check
        num_incorrect = sum(
            map(
                lambda d: (
                    d["has_answer"]
                    and "".join(d["paragraph_tokens"][d["start_index"] : d["end_index"]])
                    != d["answer_text"]
                ),
                preprocessed,
            )
        )

        logger.info(
            f"Dataset size: {len(data)} --(preprocessing)--> {len(preprocessed)}, "
            f"{num_incorrect} among them cannot correctly recover their answer text"
        )
        return preprocessed

    def __len__(self):
        return len(self.data)
