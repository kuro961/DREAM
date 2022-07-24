import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, emb_size=768, lang_num=7):
        super().__init__()
        self.meaning_emb_layer = nn.Linear(emb_size, emb_size)
        self.lang_emb_layer = nn.Linear(emb_size, emb_size)
        self.lang_iden_layer = nn.Linear(emb_size, lang_num)

    def forward(self, x):
        meaning_emb = self.meaning_emb_layer(x)
        lang_emb = self.lang_emb_layer(x)
        lang_iden = self.lang_iden_layer(lang_emb)
        return meaning_emb, lang_emb, lang_iden
