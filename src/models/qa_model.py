from torch import nn
from torch import Tensor, LongTensor
from transformers import AutoModel

from models.base import BaseModel


class QAModel(nn.Module, BaseModel):
    def __init__(
        self,
        bert_name: str = "bert-base-chinese",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.head = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids: LongTensor) -> Tensor:
        """
        Arguments
            input_ids: torch.LongTensor of shape (B, N, L)
        Returns
            start_logits: torch.FloatTensor of shape (B, N)
            end_logits: torch.FloatTensor of shape (B, N)
        """
        B, L = input_ids.shape
        bert_output = self.bert(input_ids)
        logits = self.head(bert_output.last_hidden_state)
        start_logits, end_logits = map(lambda t: t.squeeze(-1), logits.split(1, dim=-1))
        return start_logits, end_logits
