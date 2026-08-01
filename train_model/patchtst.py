import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse
import torch
import torch.nn as nn

from common import load_npz, compute_mean_std, create_loaders, train_and_evaluate, set_seed
from layers import PatchEmbedding, LearnablePositionalEmbedding


class PatchTST(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        in_channels=1,
        out_channels=1,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        num_patches = (input_len - patch_len) // stride + 1
        if num_patches <= 0:
            raise ValueError("patch_len is larger than input length")
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.patch_embed = PatchEmbedding(patch_len, stride, in_channels, d_model)
        self.pos = LearnablePositionalEmbedding(num_patches, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, pred_len * out_channels)

    def forward(self, x):
        tokens = self.patch_embed(x)
        tokens = self.pos(tokens)
        enc = self.encoder(tokens)
        pooled = enc.mean(dim=1)
        out = self.head(pooled)
        return out.view(x.size(0), self.pred_len, self.out_channels)


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
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--out_dir", default="./train/outputs/patchtst")
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

    model = PatchTST(
        input_len=input_len,
        pred_len=pred_len,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
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
