import math

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, do_cast=True):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.do_cast = do_cast

    def forward(self, x):
        # 强制转换为 float32
        if self.do_cast:
            x = x.to(torch.float32)
        # 计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        # 归一化
        x_normalized = (x - mean) / std
        return self.weight * x_normalized + self.bias

    def extra_repr(self):
        return f'dim={self.weight.shape[0]}, eps={self.eps}, do_cast={self.do_cast}'


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, do_cast=True):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.do_cast = do_cast

    def forward(self, x):
        # 强制转换为 float32
        if self.do_cast:
            x = x.to(torch.float32)
        # 计算 RMS
        # rms = torch.norm(x, p=2, dim=-1, keepdim=True) * torch.rsqrt(torch.tensor([x.size(-1)], dtype=x.dtype)) + self.eps
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

    def extra_repr(self):
        return f'dim={self.weight.shape[0]}, eps={self.eps}, do_cast={self.do_cast}'


def print_tensor_info(x: torch.Tensor, name: str):
    print(f"{name} - dtype: {x.dtype}, shape: {x.shape}, device: {x.device}")
    print(f"{name} - min: {x.min().item()}, max: {x.max().item()}")
    x_abs_max = x.abs().max().item()
    x_abs_min = x.abs().min().item()
    print(f"{name} - abs min: {x_abs_min}, abs max: {x_abs_max}")


@torch.no_grad()
def benchmark(x: torch.Tensor, layer_norm: nn.Module, rms_norm: nn.Module):
    # LayerNorm
    layer_norm_output_f32 = layer_norm(x)
    rms_norm_output_f32 = rms_norm(x)

    x_half = x.to(torch.float16)
    layer_norm_output_f16 = layer_norm(x_half)
    rms_norm_output_f16 = rms_norm(x_half)

    x_bf16 = x.to(torch.bfloat16)
    layer_norm_output_bf16 = layer_norm(x_bf16)
    rms_norm_output_bf16 = rms_norm(x_bf16)

    ln_diff_f16 = torch.abs(layer_norm_output_f32 - layer_norm_output_f16)
    rms_diff_f16 = torch.abs(rms_norm_output_f32 - rms_norm_output_f16)
    ln_diff_bf16 = torch.abs(layer_norm_output_f32 - layer_norm_output_bf16)
    rms_diff_bf16 = torch.abs(rms_norm_output_f32 - rms_norm_output_bf16)

    print_tensor_info(ln_diff_f16, "LayerNorm Diff F16")
    print_tensor_info(rms_diff_f16, "RMSNorm Diff F16")
    print_tensor_info(ln_diff_bf16, "LayerNorm Diff BF16")
    print_tensor_info(rms_diff_bf16, "RMSNorm Diff BF16")

    return rms_diff_f16.max().item()


if __name__ == "__main__":
    torch.manual_seed(42)

    for do_cast in [True, False]:
        layernorm_cast = LayerNorm(dim=1024, do_cast=do_cast).eval()
        rmsnorm_cast = RMSNorm(dim=1024, do_cast=do_cast).eval()
        layernorm_cast.weight.data = torch.randn_like(layernorm_cast.weight)
        rmsnorm_cast.weight.data = torch.randn_like(rmsnorm_cast.weight)
        layernorm_cast.bias.data = torch.randn_like(layernorm_cast.bias)

        print("=" * 50)
        print(f"Benchmarking LayerNorm and RMSNorm with do_cast={do_cast}")
        print("---")
        for i in [64, 128, 256, 512]:
            x = torch.cat([torch.randn(1, i) * 600, torch.randn(1, 1024 - i)], dim=-1)
            print_tensor_info(x, f"Input Tensor (do_cast={do_cast}, size={i})")
            benchmark(x, layernorm_cast, rmsnorm_cast)
            print("---")
        
        print("=" * 50)


        # 绘制 std 和 rms 误差分布图
        x = torch.randn(1, 1024)
        stds = [
            *list(range(1, 10, 1)),
            *list(range(10, 100, 10)),
            *list(range(100, 1000, 100)),
        ]
        rms_errors = []
        for std in stds:
            x.normal_(0, std)
            print_tensor_info(x, f"Input Tensor (do_cast={do_cast}, std={std})")
            error = benchmark(x, layernorm_cast, rmsnorm_cast)
            print("---")
            rms_errors.append(error)

        plt.figure(figsize=(10, 5))
        plt.plot(stds, rms_errors, marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input Tensor Std')
        plt.ylabel('RMSNorm Error')
        plt.title(f'RMSNorm Error vs Input Tensor Std (do_cast={do_cast})')
        plt.grid()
        plt.savefig(f'outputs/rmsnorm_error_vs_std_do_cast_{do_cast}.png')
        plt.show()
