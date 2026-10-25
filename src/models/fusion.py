import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoder, TextEncoder

class ImageOnly(nn.Module):
    def __init__(self, vocab_size=None, hidden_dim=256, num_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q=None):
        return self.head(self.img_enc(img))

class TextOnly(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2):
        super().__init__()
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        return self.head(self.txt_enc(q))

class EarlyFusion(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img, q):
        h_img = self.img_enc(img)
        h_txt = self.txt_enc(q)
        return self.head(torch.cat([h_img, h_txt], dim=-1))

class DeepFusion(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        h_img = self.img_enc(img)
        h_txt = self.txt_enc(q)
        fused = h_img * h_txt + h_img + h_txt
        return self.head(self.fusion(fused))

class LMF(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2, rank=4):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.img_factor = nn.Linear(hidden_dim, rank, bias=False)
        self.txt_factor = nn.Linear(hidden_dim, rank, bias=False)
        self.fusion_weights = nn.Linear(rank, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        h_img = self.img_enc(img)
        h_txt = self.txt_enc(q)
        img_f = self.img_factor(h_img)
        txt_f = self.txt_factor(h_txt)
        fused = self.fusion_weights(img_f * txt_f)
        return self.head(F.relu(fused))

class LateFusion(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.img_head = nn.Linear(hidden_dim, num_classes)
        self.txt_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        logits_img = self.img_head(self.img_enc(img))
        logits_txt = self.txt_head(self.txt_enc(q))
        return (logits_img + logits_txt) / 2

class CrossAttn(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2, num_heads=4):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        h_img = self.img_enc(img).unsqueeze(1)
        h_txt = self.txt_enc(q).unsqueeze(1)
        attended, _ = self.attn(h_txt, h_img, h_img)
        return self.head(attended.squeeze(1))

class GAF(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        h_img = self.img_enc(img)
        h_txt = self.txt_enc(q)
        g = self.gate(torch.cat([h_img, h_txt], dim=-1))
        fused = g * h_img + (1 - g) * h_txt
        return self.head(fused)

class HybridFusion(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=2):
        super().__init__()
        self.img_enc = ImageEncoder(hidden_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.conf_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img, q):
        h_img = self.img_enc(img)
        h_txt = self.txt_enc(q)
        weights = F.softmax(self.conf_net(torch.cat([h_img, h_txt], dim=-1)), dim=-1)
        fused = weights[:, 0:1] * h_img + weights[:, 1:2] * h_txt
        return self.head(fused)
