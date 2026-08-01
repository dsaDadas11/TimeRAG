import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQ"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(__file__))

from timemixer import TimeMixer


def ensure_array(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr


def zscore_per_series(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


class FusionDataset(Dataset):
    def __init__(self, qi, qt, tt, ts, splits, split, score_mean, score_std):
        mask = splits == split
        self.qi = qi[mask]
        self.qt = qt[mask]
        self.tt = tt[mask]
        self.ts = ts[mask]
        self.score_mean = np.asarray(score_mean).reshape(-1)
        self.score_std = np.asarray(score_std).reshape(-1)

    def __len__(self):
        return len(self.qi)

    def __getitem__(self, idx):
        qi = self.qi[idx]
        qt = self.qt[idx]
        mu = float(qi.mean())
        sd = float(qi.std())
        x_base = (qi - mu) / (sd + 1e-6)
        y = (qt - mu) / (sd + 1e-6)

        tt = self.tt[idx]
        tt_norm = (tt - mu) / (sd + 1e-6)
        if tt_norm.ndim == 3:
            if tt_norm.shape[-1] == 1:
                tt_norm = tt_norm[..., 0]
                tt_seq = np.transpose(tt_norm, (1, 0))
            else:
                tt_seq = np.transpose(tt_norm, (1, 0, 2))
                tt_seq = tt_seq.reshape(tt_seq.shape[0], -1)
        else:
            tt_seq = np.transpose(tt_norm, (1, 0))
        ts_norm = (self.ts[idx] - self.score_mean) / (self.score_std + 1e-6)
        ts_norm = np.asarray(ts_norm).reshape(-1)
        ts_rep = np.repeat(ts_norm[None, :], tt_seq.shape[0], axis=0)
        x_aux = np.concatenate([tt_seq, ts_rep], axis=1)

        if x_base.ndim == 1:
            x_base = x_base[:, None]
        if y.ndim == 1:
            y = y[:, None]

        return (
            torch.from_numpy(x_base).float(),
            torch.from_numpy(x_aux).float(),
            torch.from_numpy(y).float(),
            torch.tensor(mu),
            torch.tensor(sd),
        )


class TimeMixerSmall(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
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
        x = nn.functional.avg_pool1d(x, kernel_size=scale, stride=scale)
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


class TimeMixerFusion(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        aux_channels,
        base_scales,
        aux_scales,
        base_moving_avg=25,
        aux_moving_avg=7,
        base_dropout=0.1,
        aux_dropout=0.1,
        attn_dim=32,
        attn_heads=4,
        attn_dropout=0.1,
    ):
        super().__init__()
        if attn_dim % attn_heads != 0:
            raise ValueError("attn_dim must be divisible by attn_heads")
        self.base = TimeMixer(
            input_len=input_len,
            pred_len=pred_len,
            scales=base_scales,
            moving_avg=base_moving_avg,
            dropout=base_dropout,
        )
        self.aux = TimeMixerSmall(
            seq_len=pred_len,
            pred_len=pred_len,
            channels=aux_channels,
            scales=aux_scales,
            moving_avg=aux_moving_avg,
            dropout=aux_dropout,
        )
        self.base_proj = nn.Linear(1, attn_dim)
        self.aux_proj = nn.Linear(1, attn_dim)
        self.cross_attn = nn.MultiheadAttention(attn_dim, attn_heads, dropout=attn_dropout, batch_first=True)
        self.residual_head = nn.Linear(attn_dim, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_base, x_aux):
        base_pred = self.base(x_base)  # (B, H, 1)
        aux_pred = self.aux(x_aux).unsqueeze(-1)  # (B, H, 1)

        q = self.base_proj(base_pred)
        kv = self.aux_proj(aux_pred)
        attn_out, _ = self.cross_attn(q, kv, kv)
        delta = self.residual_head(attn_out)
        out = base_pred + self.res_scale * delta
        return out


def eval_mae_rmse(model, loader, device):
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for x_base, x_aux, y, mu, sd in loader:
            x_base = x_base.to(device)
            x_aux = x_aux.to(device)
            y = y.to(device)
            mu = mu.to(device).view(-1, 1, 1)
            sd = sd.to(device).view(-1, 1, 1)
            pred = model(x_base, x_aux)
            pred = pred * (sd + 1e-6) + mu
            y = y * (sd + 1e-6) + mu
            diff = pred - y
            mae_sum += diff.abs().sum().item()
            mse_sum += (diff ** 2).sum().item()
            count += diff.numel()
    return mae_sum / count, (mse_sum / count) ** 0.5


def train_one_epoch(model, loader, optimizer, device, scaler=None, grad_clip=None):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for batch in loader:
        x_base, x_aux, y = batch[0], batch[1], batch[2]
        x_base = x_base.to(device)
        x_aux = x_aux.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(x_base, x_aux)
            loss = loss_fn(pred, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * x_base.size(0)
    return total / len(loader.dataset)


def parse_scales(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", default="../shaanxi_cities_temperature/rag_results_all.npz")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--base_scales", default="1,2,4")
    parser.add_argument("--aux_scales", default="1,2,3")
    parser.add_argument("--base_moving_avg", type=int, default=25)
    parser.add_argument("--aux_moving_avg", type=int, default=7)
    parser.add_argument("--base_dropout", type=float, default=0.1)
    parser.add_argument("--aux_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dim", type=int, default=32)
    parser.add_argument("--attn_heads", type=int, default=1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--out_dir", default="./train/outputs/timemixer_fusion")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data_npz, allow_pickle=True)
    qi = ensure_array(data["query_inputs"]).astype(np.float32)
    qt = ensure_array(data["query_targets"]).astype(np.float32)
    tt = ensure_array(data["top_targets"]).astype(np.float32)
    ts = ensure_array(data["top_scores"]).astype(np.float32)
    splits = np.array(data["splits"])

    score_mean = ts[splits == "train"].mean(axis=0)
    score_std = ts[splits == "train"].std(axis=0)

    train_ds = FusionDataset(qi, qt, tt, ts, splits, "train", score_mean, score_std)
    val_ds = FusionDataset(qi, qt, tt, ts, splits, "val", score_mean, score_std)
    test_ds = FusionDataset(qi, qt, tt, ts, splits, "test", score_mean, score_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    input_len = qi.shape[1]
    pred_len = qt.shape[1]
    if tt.ndim == 4:
        k_feat = tt.shape[1] * tt.shape[3]
    else:
        k_feat = tt.shape[1]
    aux_channels = k_feat + ts.shape[1]

    base_scales = [s for s in parse_scales(args.base_scales) if s <= input_len]
    aux_scales = [s for s in parse_scales(args.aux_scales) if s <= pred_len]
    if not base_scales:
        base_scales = [1]
    if not aux_scales:
        aux_scales = [1]

    model = TimeMixerFusion(
        input_len=input_len,
        pred_len=pred_len,
        aux_channels=aux_channels,
        base_scales=base_scales,
        aux_scales=aux_scales,
        base_moving_avg=args.base_moving_avg,
        aux_moving_avg=args.aux_moving_avg,
        base_dropout=args.base_dropout,
        aux_dropout=args.aux_dropout,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

    best_val = float("inf")
    best_path = Path(args.out_dir) / "best.pt"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, grad_clip=args.grad_clip)
        val_mae, val_rmse = eval_mae_rmse(model, val_loader, device)
        if val_mae < best_val:
            best_val = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        scheduler.step(val_mae)
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} "
            f"lr={lr_current:.6g} early_stop={patience_cnt}/{args.patience}"
        )

        if patience_cnt >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_mae, test_rmse = eval_mae_rmse(model, test_loader, device)
    metrics = {"test_mae": test_mae, "test_rmse": test_rmse, "best_val_mae": best_val}
    with open(Path(args.out_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test MAE={test_mae:.4f} RMSE={test_rmse:.4f}")


if __name__ == "__main__":
    main()
