from torch import nn
from torch import Tensor, LongTensor
from transformers import BertModel

from models.base import BaseModel


class Selector(nn.Module, BaseModel):
    def __init__(
        self,
        bert_name: str = "bert-base-chinese",
        num_classes: int = 7,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.head = nn.Linear(512, num_classes)

    def forward(self, input_ids: LongTensor) -> Tensor:
        """
        Arguments
            input_ids: torch.LongTensor of shape (B, N, L)
        Returns
            logits: torch.FloatTensor of shape (B, N)
        """
        hidden_state = self.bert(input_ids)
        logits = self.head(hidden_state)
        return logits
