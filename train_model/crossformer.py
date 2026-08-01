import os
import sys
import math

sys.path.append(os.path.dirname(__file__))

import argparse
import torch
import torch.nn as nn

from common import load_npz, compute_mean_std, create_loaders, train_and_evaluate, set_seed


class CrossFormer(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        in_channels=1,
        out_channels=1,
        seg_len=12,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_len = input_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seg_len = seg_len
        self.num_segments = int(math.ceil(input_len / seg_len))

        self.seg_proj = nn.Linear(seg_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_segments, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dim_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, pred_len)

    def _segment(self, x):
        b, l, c = x.shape
        pad_len = self.num_segments * self.seg_len - l
        if pad_len > 0:
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x = torch.cat([x, pad], dim=1)
        x = x.view(b, self.num_segments, self.seg_len, c)
        x = x.permute(0, 3, 1, 2)  # (B, C, S, seg_len)
        return x

    def forward(self, x):
        b, l, c = x.shape
        x_seg = self._segment(x)
        # segment embedding
        x_seg = self.seg_proj(x_seg)  # (B, C, S, d_model)
        x_seg = x_seg + self.pos[:, : x_seg.size(2)].unsqueeze(1)

        # time attention per variable
        x_time = x_seg.reshape(b * c, self.num_segments, -1)
        x_time = self.time_encoder(x_time)
        x_time = x_time.view(b, c, self.num_segments, -1)

        # dimension attention per segment
        x_dim = x_time.permute(0, 2, 1, 3).contiguous()  # (B, S, C, d_model)
        x_dim = x_dim.view(b * self.num_segments, c, -1)
        x_dim = self.dim_encoder(x_dim)
        x_dim = x_dim.view(b, self.num_segments, c, -1).permute(0, 2, 1, 3)

        # aggregate segments
        rep = x_dim.mean(dim=2)  # (B, C, d_model)
        out = self.head(rep)  # (B, C, pred_len)
        return out.permute(0, 2, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", default="../shaanxi_cities_temperature/rag_results_all.npz")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--norm_mode", default="per_sample", choices=["per_sample", "global"])
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--seg_len", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="./train/outputs/crossformer")
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

    model = CrossFormer(
        input_len=input_len,
        pred_len=pred_len,
        in_channels=in_channels,
        out_channels=out_channels,
        seg_len=args.seg_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
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
