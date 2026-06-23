"""Level 2 — extract ImageNet-pretrained ViT-S/16 weights (timm used HERE ONLY).

The assignment allows loading *external pretrained tensors* into our own
``src/models/vit.py`` implementation — what is forbidden is *importing* a model
library as the model. So we use ``timm`` solely as a weight provider: create the
ImageNet-1k pretrained ViT-S/16, dump its ``state_dict()`` (tensors only, fp32)
to ``checkpoints/vit_s16_imagenet.pth``. No timm object is ever used at train time.

Source (record in report):
    timm model name = "vit_small_patch16_224.augreg_in1k"  (ImageNet-1k, augreg)

Run:
    /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/vit_extract_pretrained.py
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

TIMM_MODEL = "vit_small_patch16_224.augreg_in1k"  # ImageNet-1k pretrained (source)
OUT = Path("checkpoints/vit_s16_imagenet.pth")


def main() -> None:
    import timm  # timm imported ONLY here, for weight extraction

    print(f"timm {timm.__version__}  | creating pretrained model: {TIMM_MODEL}", flush=True)
    model = timm.create_model(TIMM_MODEL, pretrained=True)
    model.eval()

    sd = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
    print(f"extracted {len(sd)} tensors", flush=True)

    # report external key inventory (block 0 + non-block keys; blocks repeat 0..11)
    n_blocks = len({int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k))})
    print(f"\n=== external state_dict keys (block 0 + non-block; {n_blocks} blocks total) ===")
    for k in sorted(sd):
        if k.startswith("blocks.") and not k.startswith("blocks.0."):
            continue
        print(f"  {k:42s} {tuple(sd[k].shape)}")
    print(f"\npos_embed shape: {tuple(sd['pos_embed'].shape)}  cls_token: {tuple(sd['cls_token'].shape)}")

    OUT.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": sd, "timm_model": TIMM_MODEL, "source": "ImageNet-1k (timm augreg_in1k)"}, OUT)
    print(f"\nsaved -> {OUT}  (fp32, tensors only)", flush=True)


if __name__ == "__main__":
    main()
