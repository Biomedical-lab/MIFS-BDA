import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=256, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        self.proj = nn.Linear(1280, hidden_dim)

    def forward(self, x):
        feat = self.backbone(x)
        return self.proj(feat)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.gru(emb)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.proj(h)
