from typing import Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

class PatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, *input: Any, **kwargs: Any) -> Any:
        return super().forward(*input, **kwargs)

class MsgModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask) -> Any:
        with torch.no_grad():
            out = self.model(input_ids, attention_mask)
        out = out.last_hidden_state[:, 0]
        out = self.fc(out)
        return out

if __name__ == '__main__':
    m = MsgModel()