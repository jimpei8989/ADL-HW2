from typing import List

import spacy


class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load("zh_core_web_md", disable=["ner", "parser", "tagger"])

    def __call__(self, text: str) -> List[str]:
        self.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        return list(map(lambda t: t.text, self.nlp(text)))
