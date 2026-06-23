"""Model efficiency — params / FLOPs / inference FPS, used in Level 4.

params and FLOPs are hardware-independent (computed once on any device); FPS is
hardware-dependent and per the assignment must be measured on Colab T4 for
grading (H100 numbers are reference only).
"""
from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total (or trainable) parameter count."""
    ps = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in ps)


@torch.no_grad()
def count_flops(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, int, int] = (3, 224, 224),
) -> tuple[int, int]:
    """Dependency-free forward-pass MAC/FLOP counter.

    Hooks every Conv2d / Linear and sums multiply-accumulates from the *actual*
    per-layer I/O shapes (so ViT token-wise Linears are counted correctly).
    Returns ``(flops, macs)`` with ``flops ≈ 2 * macs``. Hardware-independent.
    """
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        out_hw = out.shape[2] * out.shape[3]
        k = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        macs += out_hw * m.out_channels * k

    def lin_hook(m, inp, out):
        nonlocal macs
        tokens = 1
        for s in inp[0].shape[1:-1]:   # everything between batch and feature dim
            tokens *= s
        macs += tokens * m.in_features * m.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))

    model.eval().to(device)
    x = torch.randn(1, *input_size, device=device)
    _ = model(x)
    for h in handles:
        h.remove()
    return 2 * macs, macs


@torch.no_grad()
def measure_fps(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    n_warmup: int = 20,
    n_iter: int = 200,
) -> float:
    """Measure FPS = (batch_size * n_iter) / total_time.

    Defaults follow the README spec: batch=1, 224x224, after warm-up.
    """
    model.eval().to(device)
    x = torch.randn(batch_size, *input_size, device=device)

    for _ in range(n_warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (batch_size * n_iter) / elapsed
