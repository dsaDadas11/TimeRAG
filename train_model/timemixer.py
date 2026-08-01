import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_npz, compute_mean_std, create_loaders, train_and_evaluate, set_seed
from layers import SeriesDecomp


def parse_scales(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


class TimeMixer(nn.Module):
    def __init__(self, input_len, pred_len, scales, moving_avg=25, dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.scales = scales
        self.decomp = SeriesDecomp(moving_avg)
        self.season_mlps = nn.ModuleList()
        self.trend_mlps = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        for scale in scales:
            length = max(1, input_len // scale)
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
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
        return x.permute(0, 2, 1)

    def _upsample(self, x, scale, target_len):
        if scale == 1:
            return x
        x = x.repeat_interleave(scale, dim=1)
        if x.size(1) < target_len:
            pad_len = target_len - x.size(1)
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :target_len, :]
        return x

    def _mix(self, mlp, x):
        x = x.transpose(1, 2)
        x = mlp(x)
        return x.transpose(1, 2)

    def forward(self, x):
        preds = []
        b, l, c = x.shape
        for i, scale in enumerate(self.scales):
            xs = self._downsample(x, scale)
            seasonal, trend = self.decomp(xs)
            seasonal = self._mix(self.season_mlps[i], seasonal)
            trend = self._mix(self.trend_mlps[i], trend)
            rep = seasonal + trend
            pred = self.pred_heads[i](rep.transpose(1, 2)).transpose(1, 2)
            preds.append(pred)
        weights = torch.softmax(self.scale_weights, dim=0)
        out = 0.0
        for w, p in zip(weights, preds):
            out = out + w * p
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", default="../shaanxi_cities_temperature/rag_results_all.npz")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--norm_mode", default="per_sample", choices=["per_sample", "global"])
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--out_dir", default="./train/outputs/timemixer")
    args = parser.parse_args()

    patience = args.patience if args.early_stop_patience is None else args.early_stop_patience

    set_seed(args.seed)
    data = load_npz(args.data_npz)
    inputs = data["query_inputs"]
    targets = data["query_targets"]
    input_len = inputs.shape[1]
    pred_len = targets.shape[1]
    scales = parse_scales(args.scales)

    mean_std = compute_mean_std(data)
    per_sample = args.norm_mode == "per_sample"
    train_loader, val_loader, test_loader = create_loaders(
        data, mean_std, args.batch_size, args.num_workers, per_sample_norm=per_sample, return_stats=per_sample
    )

    model = TimeMixer(
        input_len=input_len,
        pred_len=pred_len,
        scales=scales,
        moving_avg=args.moving_avg,
        dropout=args.dropout,
    )
    train_and_evaluate(
        model,
        train_loader,
        val_loader,
        test_loader,
        mean_std,
        args.out_dir,
        args.epochs,
        args.lr,
        patience,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
    )


if __name__ == "__main__":
    main()
