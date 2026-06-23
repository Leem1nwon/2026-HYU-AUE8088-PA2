"""Grad-CAM (Selvaraju et al., ICCV 2017) — multi-task aware.

We attach hooks to the last conv layer (or the final feature map of a
ViT) and compute a separate CAM for each of the three attribute heads.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Single-target Grad-CAM. Use one instance per attribute."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        score_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        """Compute a CAM for the score returned by ``score_fn``.

        Example for ``weather`` head, predicted class:

            cam = gc(x, lambda out: out["weather"].max(dim=-1).values.sum())
        """
        self.model.zero_grad()
        out = self.model(x)
        score = score_fn(out)
        score.backward(retain_graph=True)

        # Activations: (B, C, H, W). Gradients: (B, C, H, W) for CNN, or
        # (B, N, D) for ViT — students should pick the right target_layer.
        a = self._activations
        g = self._gradients
        weights = g.mean(dim=(2, 3), keepdim=True)            # (B, C, 1, 1)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))  # (B, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Per-image normalization to [0, 1].
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(1)


class ViTGradCAM:
    """Grad-CAM variant for our ViT — token features, not conv maps.

    The standard ``GradCAM`` above assumes (B, C, H, W) activations; a ViT block
    emits (B, N, D) tokens, so the channel-mean pooling does not apply. Here we
    weight each token by ``relu( (grad ⊙ act).sum over D )``, drop the CLS token,
    and reshape the remaining ``N-1`` patch tokens to the 14x14 grid. Because the
    backprop starts from a chosen head's score, each of the three attribute heads
    yields a *different* map — exactly the multi-task XAI comparison the
    assignment asks for.
    """

    def __init__(self, model: nn.Module, target_module: nn.Module | None = None,
                 grid_size: int = 14) -> None:
        self.model = model
        self.grid_size = grid_size
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        # NOTE: do NOT use the *last* block. ViT classifies from the CLS token and
        # the final LayerNorm is token-independent, so patch tokens at blocks[-1]
        # receive ~zero gradient -> the CAM is all-zero. We default to blocks[-4],
        # where patch gradients are still informative (verified empirically).
        if target_module is not None:
            target = target_module
        else:
            blocks = model.blocks
            target = blocks[max(0, len(blocks) - 4)]
        target.register_forward_hook(self._save_act)
        target.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out) -> None:
        self._act = out.detach()

    def _save_grad(self, module, grad_in, grad_out) -> None:
        self._grad = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        score_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        self.model.zero_grad()
        out = self.model(x)
        score = score_fn(out)
        score.backward(retain_graph=True)

        act, grad = self._act, self._grad         # (B, N, D)
        cam = F.relu((grad * act).sum(dim=-1))     # (B, N)
        cam = cam[:, 1:]                            # drop CLS -> (B, N-1)
        B = cam.size(0)
        g = self.grid_size
        cam = cam.reshape(B, 1, g, g)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(1)


@torch.no_grad()
def attention_rollout(
    model: nn.Module,
    x: torch.Tensor,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """Attention Rollout (Abnar & Zuidema, 2020) for our ViT.

    Recomputes each block's attention via a forward hook on its MHSA (the qkv
    Linear is reused — no edits to the backbone), fuses heads, adds the residual
    identity, row-normalizes, and multiplies across layers. Returns the
    CLS->patch attention as a (B, H, W) map in [0, 1].

    This is head-/task-agnostic (a property of the shared backbone), so it
    complements the head-specific ``ViTGradCAM``: ViTGradCAM shows where each
    *head* looks, rollout shows where the *backbone* attends overall.
    """
    attns: list[torch.Tensor] = []

    def make_hook(mhsa):
        def hook(module, inp, out):
            xin = inp[0]
            B, N, D = xin.shape
            qkv = (module.qkv(xin)
                   .reshape(B, N, 3, module.num_heads, module.head_dim)
                   .permute(2, 0, 3, 1, 4))
            q, k = qkv[0], qkv[1]
            a = (q @ k.transpose(-2, -1)) * module.scale
            attns.append(a.softmax(dim=-1).detach())   # (B, heads, N, N)
        return hook

    handles = [blk.attn.register_forward_hook(make_hook(blk.attn)) for blk in model.blocks]
    model.eval()
    _ = model(x)
    for h in handles:
        h.remove()

    B, _, N, _ = attns[0].shape
    eye = torch.eye(N, device=x.device).unsqueeze(0)
    result = eye.expand(B, N, N).clone()
    for a in attns:
        a = a.mean(dim=1) if head_fusion == "mean" else a.max(dim=1).values  # (B, N, N)
        a = a + eye                                  # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    mask = result[:, 0, 1:]                           # CLS row -> patches (B, N-1)
    g = int(round((N - 1) ** 0.5))
    mask = mask.reshape(B, 1, g, g)
    mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
    m_min = mask.amin(dim=(2, 3), keepdim=True)
    m_max = mask.amax(dim=(2, 3), keepdim=True)
    mask = (mask - m_min) / (m_max - m_min + 1e-8)
    return mask.squeeze(1)
