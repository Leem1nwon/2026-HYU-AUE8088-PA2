"""Level 2 — remap external ImageNet ViT-S/16 weights onto our src/models/vit.py.

Our ViT (``src/models/vit.py``) keys vs timm keys are *identical* except for the
MLP sub-module naming: timm uses ``mlp.fc1``/``mlp.fc2`` Linear modules, whereas
ours uses an ``nn.Sequential[Linear, GELU, Dropout, Linear, Dropout]`` so the two
Linears live at indices ``mlp.0`` and ``mlp.3``. The 1000-class ImageNet
classifier (``head.weight``/``head.bias``) has no counterpart in our multi-task
head, so it is dropped (head stays randomly initialized).

Remap rules (verified against actual timm keys, not guessed):
    blocks.{i}.mlp.fc1.{w,b}  ->  blocks.{i}.mlp.0.{w,b}
    blocks.{i}.mlp.fc2.{w,b}  ->  blocks.{i}.mlp.3.{w,b}
    head.weight, head.bias    ->  DROP
    (everything else: identity)

Run as a script for the sanity check + load report:
    /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/vit_load_pretrained.py
"""
from __future__ import annotations

from pathlib import Path

import torch

PRETRAINED_CKPT = Path("checkpoints/vit_s16_imagenet.pth")


def remap_timm_to_ours(ext_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename timm ViT keys to match our ``src/models/vit.py`` state_dict keys."""
    out: dict[str, torch.Tensor] = {}
    for k, v in ext_sd.items():
        if k in ("head.weight", "head.bias"):
            continue  # 1000-class ImageNet classifier — no counterpart in multi-task head
        nk = k
        if ".mlp.fc1." in k:
            nk = k.replace(".mlp.fc1.", ".mlp.0.")
        elif ".mlp.fc2." in k:
            nk = k.replace(".mlp.fc2.", ".mlp.3.")
        out[nk] = v
    return out


def load_pretrained_vit(model: torch.nn.Module, ckpt_path: Path = PRETRAINED_CKPT, verbose: bool = True):
    """Load remapped ImageNet weights into ``model`` (our ViT). Returns a report dict.

    Backbone is loaded; the multi-task head is left at its random init.
    """
    payload = torch.load(ckpt_path, map_location="cpu")
    ext_sd = payload["state_dict"] if "state_dict" in payload else payload
    src = payload.get("timm_model", "unknown")

    remapped = remap_timm_to_ours(ext_sd)

    # pos_embed shape sanity (token count must match: 197 = 14*14 + 1 CLS)
    model_sd = model.state_dict()
    pe_ext = tuple(remapped["pos_embed"].shape)
    pe_mdl = tuple(model_sd["pos_embed"].shape)
    pos_ok = pe_ext == pe_mdl
    if not pos_ok:
        print(f"[WARN] pos_embed shape mismatch: ext {pe_ext} vs model {pe_mdl}")

    # only keep keys that exist in the model with matching shapes
    loadable, shape_mismatch = {}, []
    for k, v in remapped.items():
        if k in model_sd:
            if tuple(model_sd[k].shape) == tuple(v.shape):
                loadable[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    result = model.load_state_dict(loadable, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    if verbose:
        print(f"source (timm) = {src}")
        print(f"pos_embed: ext {pe_ext} == model {pe_mdl} -> {pos_ok}")
        print(f"matched & loaded keys: {len(loadable)} / {len(model_sd)} model keys "
              f"({len(remapped)} remapped from {len(ext_sd)} external)")
        if shape_mismatch:
            print(f"shape mismatches (skipped): {shape_mismatch}")
        # missing_keys = model params NOT loaded -> should be ONLY the multi-task head
        print(f"missing (left at init, expect head.*): {missing}")
        print(f"unexpected (in load dict, not in model): {unexpected}")
        # confirm all backbone core params got loaded
        core = [k for k in model_sd
                if any(t in k for t in ("attn.qkv", "attn.proj", "mlp.0", "mlp.3",
                                        "norm1", "norm2", "patch_embed", "cls_token",
                                        "pos_embed")) or k in ("norm.weight", "norm.bias")]
        core_loaded = [k for k in core if k in loadable]
        print(f"backbone core params: {len(core_loaded)}/{len(core)} loaded "
              f"-> {'ALL OK' if len(core_loaded) == len(core) else 'MISSING SOME!'}")

    return {
        "n_loaded": len(loadable),
        "n_model_keys": len(model_sd),
        "missing": missing,
        "unexpected": unexpected,
        "pos_ok": pos_ok,
        "shape_mismatch": shape_mismatch,
        "source": src,
    }


def _sanity_check() -> None:
    """Forward one val batch through pretrained-loaded vs scratch model; outputs must differ."""
    from torch.utils.data import DataLoader

    from src.datasets.bdd_attr import BDDAttrDataset
    from src.models.vit import vit_small_patch16_224
    from src.utils.seed import set_seed
    from src.utils.transforms import eval_transform

    set_seed(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BDDAttrDataset("data/set_a", "val", transform=eval_transform())
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    x = batch["image"].to(device)

    scratch = vit_small_patch16_224().to(device).eval()
    pre = vit_small_patch16_224().to(device)
    rep = load_pretrained_vit(pre, verbose=True)
    pre.eval()

    with torch.no_grad():
        o_s = scratch(x)
        o_p = pre(x)
    print("\n=== sanity: scratch vs pretrained-loaded forward (val batch) ===")
    for a in ("weather", "scene", "timeofday"):
        d = (o_s[a] - o_p[a]).abs().mean().item()
        print(f"  {a}: mean|Δlogit| = {d:.4f}  (>0 => pretrained weights changed the forward)")
    print(f"\nload report: loaded={rep['n_loaded']}, missing={len(rep['missing'])}, "
          f"unexpected={len(rep['unexpected'])}, pos_ok={rep['pos_ok']}, source={rep['source']}")


if __name__ == "__main__":
    _sanity_check()
