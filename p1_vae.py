"""P1: Variational LSTM Autoencoder for regime detection."""
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import *


class SeqDataset(Dataset):
    def __init__(self, data, seq_len, targets=None):
        self.data, self.seq_len, self.targets = data, seq_len, targets
    def __len__(self): return len(self.data) - self.seq_len + 1
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx:idx+self.seq_len])
        if self.targets is not None:
            return x, torch.FloatTensor([self.targets[idx + self.seq_len - 1]])
        return x


class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)
        self.vol_head = nn.Linear(latent_dim, 1)
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        _, (h, _) = self.encoder(x)
        h = self.dropout(h.squeeze(0))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z, seq_len):
        h0 = self.to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        out, _ = self.decoder(torch.zeros(z.size(0), seq_len, self.input_dim).to(z.device), (h0, c0))
        return self.output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        vol_pred = self.vol_head(z)
        return recon, mu, logvar, z, vol_pred


def train_vae(feat, feature_cols):
    """Train VAE and extract latent features."""
    print("\n" + "=" * 60)
    print("P1: VARIATIONAL LSTM AUTOENCODER")
    print("=" * 60)

    ae_cols = [c for c in ["vix_level", "slope_3m_spot", "slope_6m_3m", "curvature",
        "contango_flag", "vix_1d_change", "vix_zscore_20d", "vix_ma_ratio",
        "RV_1d", "RV_5d", "RV_22d", "parkinson_vol"] if c in feat.columns]

    scaler = StandardScaler()
    train_raw = feat.loc[feat.index < SPLIT_DATE, ae_cols]
    scaler.fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    all_scaled = scaler.transform(feat[ae_cols])

    train_targets = feat.loc[feat.index < SPLIT_DATE, "fwd_rv_22d"].values
    train_ds = SeqDataset(train_scaled, P1_SEQ_LEN, targets=train_targets)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = LSTMVAE(len(ae_cols), P1_HIDDEN_DIM, P1_LATENT_DIM, P1_DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=P1_LR, weight_decay=1e-5)

    for epoch in range(P1_EPOCHS):
        model.train()
        r_t, kl_t, v_t, n = 0, 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            recon, mu, logvar, z, vol_pred = model(bx)
            r_loss = F.mse_loss(recon, bx)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            v_loss = F.mse_loss(vol_pred.squeeze(), by.squeeze())
            loss = r_loss + P1_KL_WEIGHT * kl_loss + P1_VOL_WEIGHT * v_loss
            opt.zero_grad(); loss.backward(); opt.step()
            r_t += r_loss.item()*bx.size(0); kl_t += kl_loss.item()*bx.size(0)
            v_t += v_loss.item()*bx.size(0); n += bx.size(0)
        if (epoch+1) % 15 == 0:
            print(f"  Epoch {epoch+1}/{P1_EPOCHS}  Recon: {r_t/n:.6f}  KL: {kl_t/n:.4f}  Vol: {v_t/n:.6f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "vae.pt"))

    # extract features
    model.eval()
    all_ds = SeqDataset(all_scaled, P1_SEQ_LEN)
    all_loader = DataLoader(all_ds, batch_size=256, shuffle=False)

    latents, recon_errors, anomaly_scores = [], [], []
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar, z, _ = model(batch)
            latents.append(mu.cpu().numpy())
            recon_errors.append(((recon - batch)**2).mean(dim=(1,2)).cpu().numpy())
            nll = F.mse_loss(recon, batch, reduction='none').mean(dim=(1,2))
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            anomaly_scores.append((nll + P1_KL_WEIGHT * kl).cpu().numpy())

    latents = np.concatenate(latents)
    dates = feat.index[P1_SEQ_LEN-1:][:len(latents)]

    latent_df = pd.DataFrame(latents, index=dates, columns=[f"latent_{i}" for i in range(P1_LATENT_DIM)])
    latent_df["recon_error"] = np.concatenate(recon_errors)
    latent_df["vae_anomaly"] = np.concatenate(anomaly_scores)

    coords = latent_df[[f"latent_{i}" for i in range(P1_LATENT_DIM)]]
    vel = np.sqrt((coords.diff()**2).sum(axis=1))
    latent_df["latent_velocity"] = vel
    latent_df["latent_acceleration"] = vel.diff()

    pre_split_crisis_years = [y for y in [2001, 2002, 2008, 2009, 2011, 2015, 2018]
                               if y < pd.Timestamp(SPLIT_DATE).year]
    crisis_mask = latent_df.index.year.isin(pre_split_crisis_years)
    if crisis_mask.sum() > 0:
        centroid = coords[crisis_mask].mean().values
        latent_df["crisis_distance"] = np.sqrt(((coords.values - centroid)**2).sum(axis=1))

    latent_df.to_csv(os.path.join(PROC_DIR, "p1_latent.csv"))
    print(f"  P1 features: {latent_df.shape}")
    return latent_df, model
