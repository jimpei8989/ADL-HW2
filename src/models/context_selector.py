from torch import nn
from torch import Tensor, LongTensor
from transformers import AutoModel

from models.base import BaseModel


class ContextSelector(nn.Module, BaseModel):
    def __init__(self, bert_name: str = "bert-base-chinese", use_pooler_output: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.head = nn.Linear(self.bert.config.hidden_size, 1)
        self.use_pooler_output = use_pooler_output

    def forward(self, input_ids: LongTensor) -> Tensor:
        """
        Arguments
            input_ids: torch.LongTensor of shape (B, N, L)
        Returns
            logits: torch.FloatTensor of shape (B, N)
        """
        bert_output = self.bert(input_ids)
        if self.use_pooler_output:
            logits = self.head(bert_output.pooler_output)
        else:
            logits = self.head(bert_output.last_hidden_state[0])
        return logits.squeeze(-1)
