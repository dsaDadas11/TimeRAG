import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse
import torch
import torch.nn as nn

from common import load_npz, compute_mean_std, create_loaders, train_and_evaluate, set_seed
from layers import SeriesDecomp, AutoCorrelationLayer


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=128, moving_avg=25, dropout=0.1, factor=3):
        super().__init__()
        self.attn = AutoCorrelationLayer(d_model, n_heads, dropout=dropout, factor=factor)
        self.decomp = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x, trend1 = self.decomp(x)
        y = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = x + y
        x, trend2 = self.decomp(x)
        return x, trend1 + trend2


class AutoformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        trend = torch.zeros_like(x)
        for layer in self.layers:
            x, t = layer(x)
            trend = trend + t
        return x, trend


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=128, moving_avg=25, dropout=0.1, factor=3):
        super().__init__()
        self.self_attn = AutoCorrelationLayer(d_model, n_heads, dropout=dropout, factor=factor)
        self.cross_attn = AutoCorrelationLayer(d_model, n_heads, dropout=dropout, factor=factor)
        self.decomp = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def _match_length(self, cross, target_len):
        if cross.size(1) == target_len:
            return cross
        if cross.size(1) > target_len:
            return cross[:, -target_len:, :]
        pad_len = target_len - cross.size(1)
        pad = cross[:, -1:, :].repeat(1, pad_len, 1)
        return torch.cat([cross, pad], dim=1)

    def forward(self, x, cross):
        cross = self._match_length(cross, x.size(1))
        x = x + self.dropout(self.self_attn(x))
        x, trend1 = self.decomp(x)
        x = x + self.dropout(self.cross_attn(x, cross, cross))
        x, trend2 = self.decomp(x)
        y = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = x + y
        x, trend3 = self.decomp(x)
        return x, trend1 + trend2 + trend3


class AutoformerDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cross, trend):
        for layer in self.layers:
            x, t = layer(x, cross)
            trend = trend + t
        return x, trend


class Autoformer(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        in_channels=1,
        out_channels=1,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        moving_avg=25,
        dropout=0.1,
        label_len=None,
        factor=3,
    ):
        super().__init__()
        if label_len is None:
            label_len = max(1, input_len // 2)
        self.label_len = min(label_len, input_len)
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(moving_avg)
        self.enc_embed = nn.Linear(in_channels, d_model)
        self.dec_embed = nn.Linear(in_channels, d_model)
        self.encoder = AutoformerEncoder(
            [AutoformerEncoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, factor) for _ in range(e_layers)]
        )
        self.decoder = AutoformerDecoder(
            [AutoformerDecoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, factor) for _ in range(d_layers)]
        )
        self.projection = nn.Linear(d_model, out_channels)

    def forward(self, x):
        b, l, c = x.shape
        seasonal, trend = self.decomp(x)
        enc_in = self.enc_embed(x)
        enc_out, _ = self.encoder(enc_in)

        trend_init = torch.cat(
            [trend[:, -self.label_len :, :], trend[:, -1:, :].repeat(1, self.pred_len, 1)], dim=1
        )
        seasonal_init = torch.cat(
            [seasonal[:, -self.label_len :, :], torch.zeros(b, self.pred_len, c, device=x.device)], dim=1
        )
        dec_in = self.dec_embed(seasonal_init)
        dec_out, dec_trend = self.decoder(dec_in, enc_out, trend_init)
        out = self.projection(dec_out) + dec_trend
        return out[:, -self.pred_len :, :]


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
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_len", type=int, default=None)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--out_dir", default="./train/outputs/autoformer")
    args = parser.parse_args()

    patience = args.patience if args.early_stop_patience is None else args.early_stop_patience

    set_seed(args.seed)
    data = load_npz(args.data_npz)
    inputs = data["query_inputs"]
    targets = data["query_targets"]
    input_len = inputs.shape[1]
    pred_len = targets.shape[1]
    in_channels = 1 if inputs.ndim == 2 else inputs.shape[2]
    out_channels = 1 if targets.ndim == 2 else targets.shape[2]

    mean_std = compute_mean_std(data)
    per_sample = args.norm_mode == "per_sample"
    train_loader, val_loader, test_loader = create_loaders(
        data, mean_std, args.batch_size, args.num_workers, per_sample_norm=per_sample, return_stats=per_sample
    )

    model = Autoformer(
        input_len=input_len,
        pred_len=pred_len,
        in_channels=in_channels,
        out_channels=out_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        moving_avg=args.moving_avg,
        dropout=args.dropout,
        label_len=args.label_len,
        factor=args.factor,
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
