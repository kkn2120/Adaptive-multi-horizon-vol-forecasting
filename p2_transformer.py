"""P2: Transformer encoder for sequential pattern matching."""
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import *


class TransDS(Dataset):
    def __init__(self, features, targets, seq_len):
        self.f, self.t, self.s = features, targets, seq_len
    def __len__(self): return len(self.f) - self.s + 1
    def __getitem__(self, idx):
        return torch.FloatTensor(self.f[idx:idx+self.s]), torch.FloatTensor([self.t[idx+self.s-1]])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2]) if d_model % 2 else torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class VolTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.norm_in = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, output_dim))

    def forward(self, x):
        h = self.norm_in(self.input_proj(x))
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.output_proj(h[:, -1, :])


def train_transformer(feat, feature_cols):
    """Train Transformer and extract embeddings."""
    print("\n" + "=" * 60)
    print("P2: TRANSFORMER")
    print("=" * 60)

    scaler = StandardScaler()
    tr_raw = feat.loc[feat.index < SPLIT_DATE, feature_cols].values
    scaler.fit(tr_raw)
    all_scaled = scaler.transform(feat[feature_cols].values)
    targets = feat["fwd_rv_22d"].values
    split_idx = (feat.index < SPLIT_DATE).sum()

    train_ds = TransDS(all_scaled[:split_idx], targets[:split_idx], P2_SEQ_LEN)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = VolTransformer(len(feature_cols), P2_D_MODEL, P2_HEADS, P2_LAYERS, P2_DROPOUT, P2_OUTPUT_DIM).to(DEVICE)
    pred_head = nn.Linear(P2_OUTPUT_DIM, 1).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(pred_head.parameters()), lr=P2_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P2_EPOCHS)

    best_loss = float('inf')
    for epoch in range(P2_EPOCHS):
        model.train(); pred_head.train()
        total, n = 0, 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = pred_head(model(bx))
            loss = F.mse_loss(pred.squeeze(), by.squeeze())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()*bx.size(0); n += bx.size(0)
        sched.step()
        avg = total/n
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer.pt"))
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{P2_EPOCHS}  Loss: {avg:.6f}")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "transformer.pt")))

    
    model.eval()
    all_ds = TransDS(all_scaled, targets, P2_SEQ_LEN)
    all_loader = DataLoader(all_ds, batch_size=256, shuffle=False)
    embeds = []
    with torch.no_grad():
        for bx, _ in all_loader:
            embeds.append(model(bx.to(DEVICE)).cpu().numpy())

    embeds = np.concatenate(embeds)
    dates = feat.index[P2_SEQ_LEN-1:][:len(embeds)]
    trans_df = pd.DataFrame(embeds, index=dates, columns=[f"trans_{i}" for i in range(P2_OUTPUT_DIM)])
    trans_df.to_csv(os.path.join(PROC_DIR, "p2_transformer.csv"))
    print(f"  P2 embeddings: {trans_df.shape}")
    return trans_df, model
