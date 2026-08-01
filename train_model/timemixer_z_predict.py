import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQ"

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


base = "../shaanxi_cities_temperature"
paths = {
    "train": f"{base}/rag_results_all_train.npz",
    "val": f"{base}/rag_results_all_val.npz",
    "test": f"{base}/rag_results_all_test.npz",
}


def zscore_per_series(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def load_split(path):
    with np.load(path, allow_pickle=True) as d:
        qt = d["query_targets"].astype(np.float32)
        tt = d["top_targets"].astype(np.float32)
        ts = d["top_scores"].astype(np.float32)
    qt_z = zscore_per_series(qt)
    tt_z = zscore_per_series(tt)
    return tt_z, ts, qt_z


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr = np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1]
    return mae, rmse, corr


def plot_samples(y_true, y_pred, split_name, out_dir):
    rng = np.random.default_rng()
    idx = rng.choice(len(y_true), size=min(6, len(y_true)), replace=False)
    plt.figure(figsize=(10, 6))
    for i, j in enumerate(idx, 1):
        plt.subplot(3, 2, i)
        plt.plot(y_true[j], label="query_z", linewidth=2)
        plt.plot(y_pred[j], label="timemixer_z", linestyle="--")
        plt.title(f"{split_name} sample {j}")
        if i == 1:
            plt.legend()
    plt.tight_layout()
    out_path = str(Path(out_dir) / f"timemixer_z_{split_name}_samples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_scatter(y_true, y_pred, split_name, corr, out_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.reshape(-1), y_pred.reshape(-1), s=6, alpha=0.3)
    plt.xlabel("query_targets (z)")
    plt.ylabel("timemixer_pred (z)")
    plt.title(f"{split_name} scatter (z): corr={corr:.3f}")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.tight_layout()
    out_path = str(Path(out_dir) / f"timemixer_z_{split_name}_scatter.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


class MovingAvg(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        trend = self.avg(x_t)
        trend = trend[..., : x_t.shape[-1]]
        trend = trend.transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class TimeMixerSmall(nn.Module):
    def __init__(
        self,
        seq_len=24,
        pred_len=24,
        channels=6,
        scales=(1, 2, 3),
        moving_avg=7,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.scales = list(scales)
        self.decomp = MovingAvg(kernel_size=moving_avg)
        self.season_mlps = nn.ModuleList()
        self.trend_mlps = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))
        self.combine = nn.Linear(channels, 1)

        for scale in self.scales:
            length = max(1, seq_len // scale)
            self.season_mlps.append(
                nn.Sequential(
                    nn.Linear(length, length),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(length, length),
                )
            )
            self.trend_mlps.append(
                nn.Sequential(
                    nn.Linear(length, length),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(length, length),
                )
            )
            self.pred_heads.append(nn.Linear(length, pred_len))

    def _downsample(self, x, scale):
        if scale == 1:
            return x
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
        return x.transpose(1, 2)

    def _mix(self, mlp, x):
        x = x.transpose(1, 2)
        x = mlp(x)
        return x.transpose(1, 2)

    def forward(self, x):
        preds = []
        for i, scale in enumerate(self.scales):
            xs = self._downsample(x, scale)
            seasonal, trend = self.decomp(xs)
            seasonal = self._mix(self.season_mlps[i], seasonal)
            trend = self._mix(self.trend_mlps[i], trend)
            rep = seasonal + trend
            out = self.pred_heads[i](rep.transpose(1, 2)).transpose(1, 2)
            out = self.combine(out).squeeze(-1)
            preds.append(out)
        if len(preds) == 1:
            return preds[0]
        weights = torch.softmax(self.scale_weights, dim=0)
        out = torch.zeros_like(preds[0])
        for w, p in zip(weights, preds):
            out = out + w * p
        return out


class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_features(tt_z, ts, ts_mean, ts_std):
    tt_seq = np.transpose(tt_z, (0, 2, 1))
    ts_norm = (ts - ts_mean) / (ts_std + 1e-6)
    ts_rep = np.repeat(ts_norm[:, None, :], tt_seq.shape[1], axis=1)
    x = np.concatenate([tt_seq, ts_rep], axis=2)
    return x


def evaluate(model, loader, device):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            gts.append(y.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    return metrics(gt, pred), pred, gt


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=200,
    lr=1e-3,
    patience=20,
    lr_decay_gamma=0.5,
    lr_decay_patience=5,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_decay_gamma,
        patience=lr_decay_patience,
        verbose=False,
    )
    loss_fn = nn.MSELoss()
    best = {"rmse": float("inf"), "state": None}
    patience_cnt = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        train_loss = total / len(train_loader.dataset)
        (mae, rmse, corr), _, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']:.6f} train_loss={train_loss:.4f} "
            f"val_mae={mae:.4f} val_rmse={rmse:.4f} val_corr={corr:.4f}"
        )
        scheduler.step(rmse)
        if rmse < best["rmse"]:
            best["rmse"] = rmse
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model


def parse_scales(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./train/outputs/timemixer_z_results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--moving_avg", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scales", default="1,2,3,4,6")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tt_tr, ts_tr, y_tr = load_split(paths["train"])
    tt_va, ts_va, y_va = load_split(paths["val"])
    tt_te, ts_te, y_te = load_split(paths["test"])

    ts_mean = ts_tr.mean(axis=0, keepdims=True)
    ts_std = ts_tr.std(axis=0, keepdims=True)

    x_tr = build_features(tt_tr, ts_tr, ts_mean, ts_std)
    x_va = build_features(tt_va, ts_va, ts_mean, ts_std)
    x_te = build_features(tt_te, ts_te, ts_mean, ts_std)

    train_loader = DataLoader(SeqDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_va, y_va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False)

    seq_len = x_tr.shape[1]
    pred_len = y_tr.shape[1]
    channels = x_tr.shape[2]

    scales = [s for s in parse_scales(args.scales) if s <= seq_len]
    if not scales:
        scales = [1]
    model = TimeMixerSmall(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        scales=scales,
        moving_avg=args.moving_avg,
        dropout=args.dropout,
    )
    model = model.to(args.device)

    model = train(
        model,
        train_loader,
        val_loader,
        args.device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        lr_decay_gamma=args.lr_decay_gamma,
        lr_decay_patience=args.lr_decay_patience,
    )
    weight_path = out_dir / "timemixer_z_best.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"saved model: {weight_path}")

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        (mae, rmse, corr), pred, gt = evaluate(model, loader, args.device)
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, corr={corr:.4f}")
        p1 = plot_samples(gt, pred, name, out_dir)
        p2 = plot_scatter(gt, pred, name, corr, out_dir)
        print(f"saved: {p1}")
        print(f"saved: {p2}")


if __name__ == "__main__":
    main()
