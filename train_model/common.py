import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_array(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "query_inputs": ensure_array(data["query_inputs"]).astype(np.float32),
        "query_targets": ensure_array(data["query_targets"]).astype(np.float32),
        "splits": np.array(data["splits"]),
    }


def compute_mean_std(data, split="train"):
    mask = data["splits"] == split
    if not mask.any():
        raise ValueError(f"split {split} not found in dataset")
    qi = data["query_inputs"][mask]
    qt = data["query_targets"][mask]
    mean_in = float(qi.mean())
    std_in = float(qi.std())
    mean_out = float(qt.mean())
    std_out = float(qt.std())
    return (mean_in, std_in), (mean_out, std_out)


class RagDataset(Dataset):
    def __init__(self, data, split, mean_std, per_sample_norm=False, return_stats=False):
        mask = data["splits"] == split
        if not mask.any():
            raise ValueError(f"split {split} not found in dataset")
        self.qi = data["query_inputs"][mask]
        self.qt = data["query_targets"][mask]
        self.per_sample_norm = per_sample_norm
        self.return_stats = return_stats or per_sample_norm
        if not per_sample_norm:
            (self.mean_in, self.std_in), (self.mean_out, self.std_out) = mean_std

    def __len__(self):
        return len(self.qi)

    def __getitem__(self, idx):
        qi = self.qi[idx]
        qt = self.qt[idx]
        if self.per_sample_norm:
            mu = float(qi.mean())
            sd = float(qi.std())
            x = (qi - mu) / (sd + 1e-6)
            y = (qt - mu) / (sd + 1e-6)
        else:
            x = (qi - self.mean_in) / (self.std_in + 1e-6)
            y = (qt - self.mean_out) / (self.std_out + 1e-6)
            mu = float(self.mean_out)
            sd = float(self.std_out)
        if x.ndim == 1:
            x = x[:, None]
        if y.ndim == 1:
            y = y[:, None]
        if self.return_stats:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.tensor(mu), torch.tensor(sd)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def create_loaders(data, mean_std, batch_size, num_workers=0, per_sample_norm=False, return_stats=False):
    train_ds = RagDataset(data, "train", mean_std, per_sample_norm=per_sample_norm, return_stats=return_stats)
    val_ds = RagDataset(data, "val", mean_std, per_sample_norm=per_sample_norm, return_stats=return_stats)
    test_ds = RagDataset(data, "test", mean_std, per_sample_norm=per_sample_norm, return_stats=return_stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, scaler=None, loss_fn=None, grad_clip=None):
    model.train()
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    total = 0.0
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(x)
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
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_mae_rmse(model, loader, device, mean_std):
    model.eval()
    mean_t, std_t = mean_std[1]
    mae_sum = 0.0
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if len(batch) >= 4:
                mu = batch[2].to(device).view(-1, 1, 1)
                sd = batch[3].to(device).view(-1, 1, 1)
                pred = pred * (sd + 1e-6) + mu
                y = y * (sd + 1e-6) + mu
            else:
                pred = pred * (std_t + 1e-6) + mean_t
                y = y * (std_t + 1e-6) + mean_t
            diff = pred - y
            mae_sum += diff.abs().sum().item()
            mse_sum += (diff ** 2).sum().item()
            count += diff.numel()
    return mae_sum / count, (mse_sum / count) ** 0.5


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    mean_std,
    out_dir,
    epochs,
    lr,
    patience,
    device=None,
    weight_decay=0.0,
    grad_clip=None,
    lr_patience=3,
    lr_factor=0.5,
    lr_min=1e-6,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
    )

    best_val = float("inf")
    best_path = out_dir / "best.pt"
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, grad_clip=grad_clip)
        val_mae, val_rmse = eval_mae_rmse(model, val_loader, device, mean_std)
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
            f"lr={lr_current:.6g} early_stop={patience_cnt}/{patience}"
        )

        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = eval_mae_rmse(model, test_loader, device, mean_std)
    metrics = {"test_mae": test_mae, "test_rmse": test_rmse, "best_val_mae": best_val}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test MAE={test_mae:.4f} RMSE={test_rmse:.4f}")
    return metrics
