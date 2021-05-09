from torch import nn
from torch import Tensor, LongTensor
from transformers import AutoModel

from models.base import BaseModel


class Selector(nn.Module, BaseModel):
    def __init__(
        self,
        bert_name: str = "bert-base-chinese",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: LongTensor) -> Tensor:
        """
        Arguments
            input_ids: torch.LongTensor of shape (B, N, L)
        Returns
            logits: torch.FloatTensor of shape (B, N)
        """
        B, N, L = input_ids.shape
        bert_output = self.bert(input_ids.reshape(-1, L))
        pooler_output = bert_output.pooler_output.reshape(B, N, -1)
        logits = self.head(pooler_output).squeeze(-1)
        return logits
