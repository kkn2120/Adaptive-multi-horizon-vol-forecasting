"""P4: Graph Attention Network for cross-asset contagion."""
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from config import *


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.15):
        super().__init__()
        self.W = nn.Linear(in_feat, out_feat, bias=False)
        self.a = nn.Linear(2*out_feat, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        h = self.W(x); N = h.size(0)
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        e = F.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1), 0.2)
        e = e.masked_fill(adj == 0, float('-inf'))
        alpha = self.dropout(torch.softmax(e, dim=1))
        return torch.matmul(alpha, h), alpha


class SimpleGAT(nn.Module):
    def __init__(self, node_feat, hidden, output, n_heads=4, dropout=0.15):
        super().__init__()
        self.heads = nn.ModuleList([GraphAttentionLayer(node_feat, hidden, dropout) for _ in range(n_heads)])
        self.pool = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*n_heads, output))
    def forward(self, x, adj):
        outs, alphas = [], []
        for head in self.heads:
            o, a = head(x, adj); outs.append(o); alphas.append(a)
        multi = torch.cat(outs, dim=-1).mean(dim=0)
        return self.pool(multi), alphas


def train_gat(feat, cross_assets_df):
    """Train GAT and extract embeddings."""
    print("\n" + "=" * 60)
    print("P4: GRAPH ATTENTION NETWORK")
    print("=" * 60)

    cross_returns = np.log(cross_assets_df / cross_assets_df.shift(1)).dropna()
    available = [c for c in cross_returns.columns if cross_returns[c].notna().sum() > 500]
    cross_returns = cross_returns[available].dropna()
    print(f"  Assets: {len(available)}")

    
    asset_features = {}
    for asset in available:
        af = pd.DataFrame(index=cross_returns.index)
        af[f"{asset}_ret22"] = cross_returns[asset].rolling(22).sum()
        af[f"{asset}_vol22"] = cross_returns[asset].rolling(22).std() * np.sqrt(252)
        af[f"{asset}_vol5"] = cross_returns[asset].rolling(5).std() * np.sqrt(252)
        asset_features[asset] = af

    # rolling correlations (every 5 days for speed)
    corr_dates = cross_returns.index[P4_CORR_WINDOW:]
    rolling_corrs = {}
    data_matrix = cross_returns.values
    print("  Computing correlations...")
    for i in range(0, len(corr_dates), 5):
        window = data_matrix[max(0,i):i+P4_CORR_WINDOW]
        if len(window) >= P4_CORR_WINDOW:
            corr = np.corrcoef(window.T)
            for j in range(5):
                if i+j < len(corr_dates):
                    rolling_corrs[corr_dates[i+j]] = corr

    
    common = sorted(set(feat.index) & set(corr_dates) & set(cross_returns.index))
    gat_data = []
    for d in common:
        if d not in rolling_corrs: continue
        nf, valid = [], True
        for asset in available:
            af = asset_features[asset]
            if d not in af.index: valid = False; break
            vals = [af.loc[d, f"{asset}_ret22"], af.loc[d, f"{asset}_vol22"], af.loc[d, f"{asset}_vol5"]]
            if any(np.isnan(v) for v in vals): valid = False; break
            nf.append(vals)
        if not valid: continue
        gat_data.append({"date": d, "nodes": np.array(nf), "adj": np.abs(rolling_corrs[d]),
                          "target": feat.loc[d, "fwd_rv_22d"]})

    split = next(i for i, g in enumerate(gat_data) if g["date"] >= pd.Timestamp(SPLIT_DATE))
    print(f"  Graph data: {len(gat_data)}, train: {split}")

    # train with shuffled batches and gradient clipping
    model = SimpleGAT(3, P4_HIDDEN, P4_OUTPUT, P4_HEADS, P4_DROPOUT).to(DEVICE)
    head = nn.Linear(P4_OUTPUT, 1).to(DEVICE)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=P4_LR, weight_decay=1e-5)

    best_loss, patience = float('inf'), 0
    for epoch in range(P4_EPOCHS):
        model.train(); head.train()
        indices = list(range(split))
        np.random.shuffle(indices)
        total, n = 0, 0
        for idx in indices:
            s = gat_data[idx]
            x = torch.FloatTensor(s["nodes"]).to(DEVICE)
            adj = torch.FloatTensor(s["adj"]).to(DEVICE)
            target = torch.FloatTensor([s["target"]]).to(DEVICE)
            embed, _ = model(x, adj)
            loss = F.mse_loss(head(embed), target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1

        avg = total/n
        if avg < best_loss:
            best_loss = avg; patience = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gat.pt"))
        else:
            patience += 1
        if patience >= P4_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}"); break
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}  Loss: {avg:.6f}")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gat.pt")))

    
    model.eval(); head.eval()
    embeds, dates_out = [], []
    with torch.no_grad():
        for i, s in enumerate(gat_data):
            x = torch.FloatTensor(s["nodes"]).to(DEVICE)
            adj = torch.FloatTensor(s["adj"]).to(DEVICE)
            embed, _ = model(x, adj)
            embeds.append(embed.cpu().numpy())
            dates_out.append(s["date"])
            if (i+1) % 500 == 0:
                print(f"  Extracting: {i+1}/{len(gat_data)}")

    gat_df = pd.DataFrame(np.array(embeds), index=dates_out, columns=[f"gat_{i}" for i in range(P4_OUTPUT)])
    gat_df["mean_abs_corr"] = [np.abs(rolling_corrs.get(d, np.eye(1))).mean() for d in dates_out]
    gat_df.to_csv(os.path.join(PROC_DIR, "p4_gat.csv"))
    print(f"  P4 embeddings: {gat_df.shape}")
    return gat_df, model
