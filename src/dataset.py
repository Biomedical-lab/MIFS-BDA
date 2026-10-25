import re
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

def simple_tokenize(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def load_vqarad(data_dir, max_len=32, seed=42):
    data_dir = Path(data_dir)
    img_dir = data_dir / "VQA_RAD_Image_Folder"

    with open(data_dir / "VQA_RAD Dataset Public.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        ans = str(r.get("answer", "")).strip().lower()
        atype = str(r.get("answer_type", "")).strip().upper()
        if atype != "CLOSED" or ans not in ("yes", "no"):
            continue
        img_path = img_dir / r["image_name"]
        if not img_path.exists():
            continue
        rows.append({
            "image_name": r["image_name"],
            "img_path": str(img_path),
            "question": r["question"],
            "y": 1 if ans == "yes" else 0,
        })

    df = pd.DataFrame(rows)

    all_tokens = []
    for q in df["question"]:
        all_tokens.extend(simple_tokenize(q))
    counts = Counter(all_tokens)
    vocab = ["<PAD>", "<UNK>"] + [t for t, c in counts.most_common() if c >= 2]
    token2idx = {t: i for i, t in enumerate(vocab)}

    def encode(text):
        tokens = simple_tokenize(text)[:max_len]
        ids = [token2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    df["question_ids"] = df["question"].apply(encode)

    imgs = df["image_name"].unique()
    train_imgs, temp_imgs = train_test_split(imgs, test_size=0.3, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed)

    train_df = df[df["image_name"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["image_name"].isin(val_imgs)].reset_index(drop=True)
    test_df = df[df["image_name"].isin(test_imgs)].reset_index(drop=True)

    return train_df, val_df, test_df, len(vocab)

class VQADataset(Dataset):
    def __init__(self, df, transform=None, mask_p=0.0):
        self.df = df
        self.transform = transform
        self.mask_p = mask_p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.mask_p > 0:
            _, h, w = img.shape
            mask = torch.rand(1, h // 16, w // 16) > self.mask_p
            mask = F.interpolate(mask.unsqueeze(0).float(), size=(h, w), mode="nearest").squeeze(0)
            img = img * mask

        q_ids = torch.tensor(row["question_ids"], dtype=torch.long)
        label = torch.tensor(row["y"], dtype=torch.long)
        return img, q_ids, label

def get_transforms(train=True, img_size=224):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
