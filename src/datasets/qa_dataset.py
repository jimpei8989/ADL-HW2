import torch
from transformers import BertTokenizer

from datasets.base import BaseDataset


class QADataset(BaseDataset):
    TO_BE_PADDED = ["input_ids"]

    def __init__(self, *args, filter_no_answer: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        if filter_no_answer:
            self.data = list(filter(lambda d: d["has_answer"], self.data))

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
        d = self.data[index]
        question_tokens = d.get("question_tokens")
        paragraph_tokens = d.get("paragraph_tokens")

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
                d.get("start_index") + len(question_tokens) + 2 if d.get("has_answer") else 0
            ),
            "end_index": (
                d.get("end_index") + len(question_tokens) + 2 if d.get("has_answer") else 0
            ),
        }


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    dataset = QADataset(
        [
            "在19世紀雙星觀測所獲得的成就使重要性也增加了。在1834年，白塞爾觀測到天狼星自行的變化，因而推測有一顆隱藏的伴星；愛德華·皮克林在1899年觀測開陽週期性分裂的光譜線時發現第一顆光譜雙星，週期是104天。天文學家斯特魯維和舍本·衛斯里·伯納姆仔細的觀察和收集了許多聯星的資料，使得可以從被確定的軌道要素推算出恆星的質量。第一個獲得解答的是1827年由菲利克斯·薩瓦里透過望遠鏡的觀測得到的聯星軌道。對恆星的科學研究在20世紀獲得快速的進展，相片成為天文學上很有價值的工具。卡爾·史瓦西發現經由比較視星等和攝影星等的差別，可以得到恆星的顏色和它的溫度。1921年，光電光度計的發展可以在不同的波長間隔上非常精密的測量星等。阿爾伯特·邁克生在虎克望遠鏡第一次使用干涉儀測量出恆星的直徑。",  # noqa: E501
            "在19世紀雙星觀測所獲得的成就使重要性也增加了。",
        ],
        [
            {
                "id": "ab39567999fd376480ac3076904e598e",
                "question": "舍本和誰的數據能推算出連星的恆星的質量？",
                "paragraphs": [0, 1],
                "relevant": 0,
                "answers": [{"text": "斯特魯維", "start": 108}],
            },
        ],
        tokenizer,
    )

    d = dataset[0]

    print(tokenizer.convert_ids_to_tokens(d["input_ids"][d["start_index"] : d["end_index"]]))
