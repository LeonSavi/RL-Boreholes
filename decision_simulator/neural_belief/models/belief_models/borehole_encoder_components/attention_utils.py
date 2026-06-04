"""Attention extraction and visualization for VariableAwarePatchBoreholeTransformerEncoder.

Usage
-----
    latent, attn_weights = encoder(x, return_attention=True)
    # attn_weights: (n_layers, B, n_heads, seq_len, seq_len)

    cls_attn = extract_cls_attention(attn_weights, n_variables=V, n_patches=P)
    # cls_attn: (n_layers, B, n_heads, n_variables, n_patches)

    ax = plot_cls_attention(cls_attn, var_names=["Fe", "Al", ...])
"""

from __future__ import annotations

import torch


def extract_cls_attention(
    attn_weights: torch.Tensor,
    n_variables: int,
    n_patches: int,
) -> torch.Tensor:
    """Extract and reshape CLS-token attention to variable × patch layout.

    Parameters
    ----------
    attn_weights : (n_layers, B, n_heads, seq_len, seq_len)
        Raw attention weights from VariableAwarePatchBoreholeTransformerEncoder
        called with return_attention=True.
        seq_len = 1 + n_variables * n_patches  (CLS token first).
    n_variables  : number of geological variables V
    n_patches    : number of depth patches P

    Returns
    -------
    cls_attn : (n_layers, B, n_heads, n_variables, n_patches)
        Attention from the CLS token to each (variable, depth-patch) token.
    """
    # Row 0 is the CLS query; columns 1: are the variable-patch tokens.
    cls_attn = attn_weights[:, :, :, 0, 1:]  # (n_layers, B, n_heads, V * P)
    return cls_attn.reshape(*cls_attn.shape[:-1], n_variables, n_patches)


def plot_cls_attention(
    cls_attn: torch.Tensor,
    layer_idx: int = -1,
    sample_idx: int = 0,
    var_names: list[str] | None = None,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Visualize mean CLS attention over heads as a geological-variable × depth-patch heatmap.

    Parameters
    ----------
    cls_attn   : (n_layers, B, n_heads, n_variables, n_patches)
                 Output of extract_cls_attention().
    layer_idx  : which transformer layer to visualize; -1 = last layer.
    sample_idx : which element of the batch to visualize.
    var_names  : geological variable names for the y-axis. Falls back to
                 "Var 0", "Var 1", … when None.
    title      : plot title. Auto-generated when None.
    ax         : existing Axes to draw on; creates a new figure when None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the heatmap. The colorbar is attached automatically
        when a new figure is created.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    layer_attn = cls_attn[layer_idx, sample_idx]  # (n_heads, V, P)
    mean_attn = layer_attn.mean(dim=0)             # (V, P)
    n_variables, n_patches = mean_attn.shape

    if var_names is None:
        var_names = [f"Var {v}" for v in range(n_variables)]

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(
            figsize=(max(6, n_patches * 0.6), max(3, n_variables * 0.5))
        )

    data = mean_attn.detach().cpu().numpy()
    im = ax.imshow(data, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(n_patches))
    ax.set_xticklabels([str(p) for p in range(n_patches)], fontsize=8)
    ax.set_yticks(np.arange(n_variables))
    ax.set_yticklabels(var_names, fontsize=8)
    ax.set_xlabel("Depth patch")
    ax.set_ylabel("Geological variable")

    n_layers = cls_attn.shape[0]
    layer_label = f"layer {layer_idx % n_layers}"
    n_heads = layer_attn.shape[0]
    if title is None:
        title = f"CLS attention — {layer_label}  (mean over {n_heads} heads)"
    ax.set_title(title)

    if create_fig:
        plt.colorbar(im, ax=ax, label="attention weight")
        plt.tight_layout()

    return ax
