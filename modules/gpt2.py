import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F
from utils import load_file

class GPT2(nn.Module):
    def __init__(self, indicator=8,):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        self.n_hidden = self.encoder.config.hidden_size
        self.prediction = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden // 2),
            nn.Tanh(),
            nn.Linear(self.n_hidden // 2, 1),
        )
        self.node_indicator = indicator

    def forward(self, inputs):
        [seq, label] = inputs
        representations = self.encoder(seq)[0]
        mask_idx = torch.eq(seq, self.node_indicator)
        hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
            -1, self.n_hidden)
        answer_logit = self.prediction(hidden).squeeze(1)
        if label is None:
            return torch.sigmoid(answer_logit)
        return F.binary_cross_entropy_with_logits(answer_logit, label