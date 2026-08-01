import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x):
        pad = self.kernel_size - 1
        pad_left = pad // 2
        pad_right = pad - pad_left
        x = x.permute(0, 2, 1)
        x = F.pad(x, (pad_left, pad_right), mode="replicate")
        x = self.avg(x)
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=3):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = q
        b, l, d = q.shape
        q = self.q_proj(q).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        orig_dtype = q.dtype
        if q.dtype in (torch.float16, torch.bfloat16):
            q = q.float()
            k = k.float()
            v = v.float()

        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=l, dim=2)

        corr_mean = corr.mean(dim=-1)
        top_k = max(1, int(self.factor * math.log(l)))
        top_k = min(top_k, l)
        top_idx = torch.topk(corr_mean, top_k, dim=-1).indices
        weights = torch.softmax(corr_mean.gather(-1, top_idx), dim=-1)

        base = torch.arange(l, device=q.device)
        out = torch.zeros_like(v)
        for i in range(top_k):
            delay = top_idx[..., i]
            idx = (base.view(1, 1, l) + delay.unsqueeze(-1)) % l
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
            v_shift = v.gather(2, idx)
            w = weights[..., i].unsqueeze(-1).unsqueeze(-1)
            out = out + v_shift * w

        out = out.transpose(1, 2).contiguous().view(b, l, d)
        out = self.out_proj(out)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return self.dropout(out)


class FourierBlock(nn.Module):
    def __init__(self, d_model, modes=32):
        super().__init__()
        self.modes = modes
        self.weight_real = nn.Parameter(torch.randn(modes, d_model) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(modes, d_model) * 0.02)

    def forward(self, x):
        b, l, d = x.shape
        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        x_ft = torch.fft.rfft(x, dim=1)
        m = min(self.modes, x_ft.size(1))
        weight = torch.complex(self.weight_real[:m], self.weight_imag[:m])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :m, :] = x_ft[:, :m, :] * weight
        out = torch.fft.irfft(out_ft, n=l, dim=1)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, in_channels, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self.proj = nn.Linear(patch_len * in_channels, d_model)

    def forward(self, x):
        b, l, c = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(b, patches.size(1), self.patch_len * c)
        return self.proj(patches)


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos[:, : x.size(1)]
