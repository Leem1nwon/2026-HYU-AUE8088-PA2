"""Class-balanced samplers for the multi-task setting.

Multi-task Imbalance is *not* a solved problem — students must decide
which attribute to balance against (or design a hybrid). The helper
below balances against a single attribute. Extending it to a joint
balancing scheme is part of Level 3.
"""
from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from .bdd_attr import ATTRIBUTES, BDDAttrDataset


def class_balanced_sampler(
    dataset: BDDAttrDataset,
    attribute: str = "weather",
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Inverse-frequency sampling over a single attribute."""
    counts = dataset.class_counts(attribute).float()
    # Avoid division by zero for absent classes.
    inv_freq = 1.0 / counts.clamp(min=1)

    weights = torch.zeros(len(dataset))
    for i, s in enumerate(dataset.samples):
        label = getattr(s, attribute)
        if label >= 0:
            weights[i] = inv_freq[label]

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )


def joint_class_balanced_sampler(
    dataset: BDDAttrDataset,
    attributes: tuple[str, ...] = ATTRIBUTES,
    mode: str = "mean",
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Joint multi-task balancing — the Level 3 extension.

    A single sampler cannot balance all three attributes at once (upsampling
    ``snowy`` for *weather* skews the *scene*/*timeofday* mix of what gets drawn).
    The conflict-handling scheme here: per attribute, compute each sample's
    inverse-frequency weight and normalize so its mean over the dataset is 1
    (puts the three attributes on a comparable scale); then combine the three
    per-sample weights.

        mode="mean"    -> average of the 3 normalized weights
                          (a sample rare in *any* attribute is upsampled)
        mode="product" -> product of the 3 (aggressively favors jointly-rare
                          combinations, e.g. snowy+night)

    Absent classes (``foggy`` has 0 train samples) never occur in the labels, so
    they contribute no weight — the sampler cannot conjure samples that do not
    exist (a limitation worth noting in the report).
    """
    per_attr_w: dict[str, torch.Tensor] = {}
    for a in attributes:
        counts = dataset.class_counts(a).float()
        inv = 1.0 / counts.clamp(min=1)
        # per-sample weight normalized to mean 1 over the labeled dataset
        sample_w = torch.tensor(
            [inv[getattr(s, a)].item() if getattr(s, a) >= 0 else 0.0 for s in dataset.samples]
        )
        mean = sample_w[sample_w > 0].mean().clamp(min=1e-8)
        per_attr_w[a] = sample_w / mean

    weights = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        vals = [per_attr_w[a][i].item() for a in attributes if dataset.samples[i].__getattribute__(a) >= 0]
        if not vals:
            continue
        if mode == "product":
            w = 1.0
            for v in vals:
                w *= v
            weights[i] = w
        else:  # "mean"
            weights[i] = sum(vals) / len(vals)

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )
