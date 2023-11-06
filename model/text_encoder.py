from transformers import AutoModel
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.model = AutoModel.from_pretrained(llm)

    def forward(self, **inputs):
        return self.model(**inputs)
