from itertools import chain
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from datasets.dataset_utils import split_context_and_tokenize, NO_SPAN
from utils.io import json_load
from utils.logger import logger
from utils.tqdmm import tqdmm


class QADataset(Dataset):
    @classmethod
    def from_json(cls, context_json: Path, data_json: Path, **kwargs):
        contexts = json_load(context_json)
        data = json_load(data_json)
        return cls(contexts, data, **kwargs)

    def __init__(
        self,
        contexts: List[str],
        data: List[dict],
        tokenizer: Optional[BertTokenizer] = None,
        test: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.test = test

        self.preprocess_dataset(contexts, data)

    def preprocess_dataset(self, contexts, data):
        self.data = list(
            chain.from_iterable(
                split_context_and_tokenize(
                    d["question"],
                    contexts[d["relevant"]],
                    d["answers"][0]["start"],
                    d["answers"][0]["start"] + len(d["answers"][0]["text"]),
                    self.tokenizer,
                    add_to_dict={
                        "id": d["id"],
                        "answer_text": d["answers"][0]["text"],
                    },
                )
                for d in tqdmm(data, desc="Preprocessing dataset for QA")
            )
        )

        # Sanity check
        num_incorrect = sum(
            map(
                lambda d: (
                    d["start_index"] != NO_SPAN
                    and "".join(d["paragraph_tokens"][d["start_index"] : d["end_index"]])
                    != d["answer_text"]
                ),
                self.data,
            )
        )

        logger.info(
            f"Dataset size: {len(data)} --(preprocessing)--> {len(self.data)}, "
            f"{num_incorrect} among them cannot correctly recover their answer text"
        )

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
        question_tokens = self.data[index].get("question_tokens")
        paragraph_tokens = self.data[index].get("paragraph_tokens")

        input_ids = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token]
            + question_tokens
            + [self.tokenizer.sep_token]
            + paragraph_tokens
            + [self.tokenizer.sep_token]
        )

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "start_index": (
                0
                if self.data[index].get("start_index") == NO_SPAN
                else self.data[index].get("start_index") + len(question_tokens) + 2
            ),
            "end_index": (
                0
                if self.data[index].get("end_index") == NO_SPAN
                else self.data[index].get("end_index") + len(question_tokens) + 2
            ),
        }


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    dataset = QADataset(
        [
            "在19世紀雙星觀測所獲得的成就使重要性也增加了。在1834年，白塞爾觀測到天狼星自行的變化，因而推測有一顆隱藏的伴星；愛德華·皮克林在1899年觀測開陽週期性分裂的光譜線時發現第一顆光譜雙星，週期是104天。天文學家斯特魯維和舍本·衛斯里·伯納姆仔細的觀察和收集了許多聯星的資料，使得可以從被確定的軌道要素推算出恆星的質量。第一個獲得解答的是1827年由菲利克斯·薩瓦里透過望遠鏡的觀測得到的聯星軌道。對恆星的科學研究在20世紀獲得快速的進展，相片成為天文學上很有價值的工具。卡爾·史瓦西發現經由比較視星等和攝影星等的差別，可以得到恆星的顏色和它的溫度。1921年，光電光度計的發展可以在不同的波長間隔上非常精密的測量星等。阿爾伯特·邁克生在虎克望遠鏡第一次使用干涉儀測量出恆星的直徑。"
        ],
        [
            {
                "id": "ab39567999fd376480ac3076904e598e",
                "question": "舍本和誰的數據能推算出連星的恆星的質量？",
                "relevant": 0,
                "answers": [{"text": "斯特魯維", "start": 108}],
            }
        ],
        tokenizer,
    )

    d = dataset[0]

    print(tokenizer.convert_ids_to_tokens(d["input_ids"][d["start_index"] : d["end_index"]]))
