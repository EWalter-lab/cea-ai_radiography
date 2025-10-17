"""SEA utilities for cost-effectiveness analysis (CEA).

This module bundles small, well-typed helpers you can reuse in
**stochastic (sampling) uncertainty / SEA** workflows:

- `confidence_ellipse`: draw a (Gaussian) confidence ellipse for a cloud of
  (ΔEffectiveness, ΔCost) points on a Matplotlib axis.
- `calculate_delta_cost_effectiveness`: compute per-group deltas between
  AI=1 and AI=0 arms from a tidy table with cost/effect columns.
- `calculate_icer`: incremental cost-effectiveness ratio helper.
- `bootstrap_samples`: bootstrap ΔCost/ΔEffect pairs under several schemes
  (standard / balanced / weighted) with a reproducible RNG.

Quick start
-----------
Given a tidy table with per-examiner costs/effects for AI on/off:

>>> import pandas as pd
>>> from matplotlib import pyplot as plt
>>> from cea_radiography import evaluator
>>>
>>> # df columns (example): examiner, experience_level, ai_diagnosis, TP_TDcost, TP_TDeffectiveness
>>> deltas = evaluator.calculate_delta_cost_effectiveness(
...     df,
...     cost_col="TP_TDcost",
...     eff_col="TP_TDeffectiveness",
...     group_cols=("examiner", "experience_level"),
...     ai_col="ai_diagnosis",
... )
>>>
>>> boots = evaluator.bootstrap_samples(
...     deltas,
...     columns_cost="delta_cost",
...     columns_effectiveness="delta_effectiveness",
...     n=2000,
...     mode="balanced",
...     group_col="experience_level",
...     random_state=42,
... )
>>>
>>> fig, ax = plt.subplots()
>>> ax.scatter(
...     boots["delta_effectiveness"],
...     boots["delta_cost"],
...     s=8,
...     alpha=0.2,
... )
>>> evaluator.confidence_ellipse(
...     x=boots["delta_effectiveness"].to_numpy(),
...     y=boots["delta_cost"].to_numpy(),
...     ax=ax,
...     n_std=1.96,
...     edgecolor="black",
...     linewidth=1.0,
...     facecolor="none",
... )
>>> ax.axhline(0, linewidth=1)
>>> ax.axvline(0, linewidth=1)
>>> ax.set_xlabel("Δ Effectiveness")
>>> ax.set_ylabel("Δ Cost")
>>> fig.tight_layout()

Notes
-----
- Ellipses assume approximate bivariate normality
- For ICER summaries, combine `calculate_icer` with the bootstrapped sample
  means or directly with arm means.
"""

from __future__ import annotations


from typing import Optional, Sequence, Tuple, Mapping, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

plt.rcParams["font.family"]= "Arial"
plt.rcParams["font.size"]= 12
plt.rcParams["svg.fonttype"] = "none"

def confidence_ellipse(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    ax: Axes,
    *,
    n_std: float = 1.96,
    facecolor: str = "none",
    **kwargs,
) -> Ellipse:
    """Add a Gaussian confidence ellipse for paired data to `ax`.

    The ellipse is centered at (mean(x), mean(y)) with radii derived from the
    eigenvalues of the sample covariance matrix. If the covariance is singular
    (e.g., all points identical), the ellipse collapses to a line/point.

    Args:
        x: 1D array-like of x-values (e.g., ΔEffectiveness).
        y: 1D array-like of y-values (e.g., ΔCost).
        ax: Matplotlib axes to draw on.
        n_std: Number of standard deviations for the ellipse radii.
            `1.96` ~ 95% CI if the cloud is approximately Gaussian.
        facecolor: Fill color for the ellipse. Defaults to "none".
        **kwargs: Extra `matplotlib.patches.Ellipse` keyword arguments
            (e.g., `edgecolor`, `linewidth`, `alpha`).

    Returns:
        The added `Ellipse` patch.

    Raises:
        ValueError: If `x` and `y` do not have the same length after NaN filtering.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must be the same size")
    if x_arr.size < 2:
        mean_x = float(np.mean(x_arr)) if x_arr.size else 0.0
        mean_y = float(np.mean(y_arr)) if y_arr.size else 0.0
        ellipse = Ellipse((mean_x, mean_y), width=0.0, height=0.0, angle=0.0, facecolor=facecolor, **kwargs)
        return ax.add_patch(ellipse)

    cov = np.cov(np.stack([x_arr, y_arr], axis=0))
    mean_x, mean_y = float(np.mean(x_arr)), float(np.mean(y_arr))

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width, height = 2.0 * n_std * np.sqrt(np.maximum(eigvals, 0.0))
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor=facecolor,
        **kwargs,
    )
    return ax.add_patch(ellipse)


def calculate_delta_cost_effectiveness(
    df: pd.DataFrame,
    *,
    cost_col: str = "cost_per_case",
    eff_col: str = "effectiveness_per_case",
    group_cols: Sequence[str] = ("examiner", "experience_level"),
    ai_col: str = "ai_diagnosis",
) -> pd.DataFrame:
    """Compute ΔCost and ΔEffectiveness (AI=1 minus AI=0) per group.

    The input is a tidy table with a binary arm column (e.g., `ai_diagnosis`)
    and numeric `cost_col` / `eff_col`. For each `group_cols` combination,
    this function pivots arms to columns and subtracts (1 − 0).

    Args:
        df: Input DataFrame.
        cost_col: Column with cost values.
        eff_col: Column with effectiveness values.
        group_cols: Keys that define each group (e.g., examiner stratification).
        ai_col: Column indicating the arm (must contain 0 and 1 within groups).

    Returns:
        DataFrame with columns:
            - all `group_cols`
            - `delta_cost`
            - `delta_effectiveness`

        Groups missing either arm (0 or 1) are dropped.

    Raises:
        KeyError: If any required columns are missing.
    """
    need = [*group_cols, ai_col, cost_col, eff_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    pivot = df.pivot_table(
        index=list(group_cols),
        columns=ai_col,
        values=[cost_col, eff_col],
        aggfunc="mean",
        dropna=False,
    )

    has_arm0 = pivot[(cost_col, 0)].notna() & pivot[(eff_col, 0)].notna()
    has_arm1 = pivot[(cost_col, 1)].notna() & pivot[(eff_col, 1)].notna()
    valid = pivot[has_arm0 & has_arm1].copy()

    delta_cost = valid[(cost_col, 1)] - valid[(cost_col, 0)]
    delta_eff = valid[(eff_col, 1)] - valid[(eff_col, 0)]

    out = valid.index.to_frame(index=False)
    out["delta_cost"] = delta_cost.to_numpy()
    out["delta_effectiveness"] = delta_eff.to_numpy()
    return out


def calculate_icer(
    cost_baseline: float,
    eff_baseline: float,
    cost_new: float,
    eff_new: float,
    *,
    scale_effectiveness: float = 1.0,
) -> float:
    """Compute the incremental cost-effectiveness ratio (ICER).

    ICER(new vs baseline) = (C_new − C_base) / (E_new − E_base).

    Args:
        cost_baseline: Baseline arm cost.
        eff_baseline: Baseline arm effectiveness.
        cost_new: New/intervention arm cost.
        eff_new: New/intervention arm effectiveness.
        scale_effectiveness: Optional multiplicative scale for both effectiveness
            values (useful to convert units). Default is 1.0.

    Returns:
        The ICER as a float, or `np.nan` if the incremental effectiveness is 0.
    """
    e0 = eff_baseline * scale_effectiveness
    e1 = eff_new * scale_effectiveness
    delta_c = cost_new - cost_baseline
    delta_e = e1 - e0
    return float(delta_c / delta_e) if delta_e != 0 else float(np.nan)


def bootstrap_samples(
    df: pd.DataFrame,
    *,
    columns_cost: str = "delta_cost",
    columns_effectiveness: str = "delta_effectiveness",
    n: int = 1000,
    label: str | None = None,
    mode: str = "standard",
    weights: Mapping[object, float] | None = None,
    group_col: str = "experience_level",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap ΔCost/ΔEffect pairs for SEA.

    Supports three schemes:
      - **standard**: sample `len(df)` rows with replacement from the whole table.
      - **balanced**: sample the same number of rows per `group_col`; the per-group
        sample size is `min(group sizes)`. This guards against dominance of large
        groups.
      - **weighted**: draw rows with replacement using row probabilities derived
        from a dict of group weights, mapping `group_col` levels -> weight.

    Notes:
        - Sampling is implemented with NumPy's `Generator.choice` for speed and
          reproducibility. Set `random_state` for deterministic results.
        - This function returns the **bootstrap distribution of the mean** ΔCost
          and ΔEffectiveness (one point per replication). If you need to retain
          the full resampled rows per replication, adapt the body to store them.

    Args:
        df: Input DataFrame containing at least `columns_cost` and `columns_effectiveness`.
        columns_cost: Column with ΔCost values.
        columns_effectiveness: Column with ΔEffectiveness values.
        n: Number of bootstrap replications.
        label: Optional constant label to attach (e.g., a stratum name).
        mode: One of {"standard", "balanced", "weighted"}.
        weights: For `mode="weighted"`, mapping from `group_col` level to relative
            weight (non-negative). Levels not present in the mapping receive weight 0.
        group_col: Column that defines groups for `balanced`/`weighted` modes.
        random_state: Seed for NumPy's random number generator.

    Returns:
        DataFrame with columns:
          - "delta_cost"
          - "delta_effectiveness"
          - plus `group_col` if `label` is provided (constant per row).

    Raises:
        KeyError: If required columns are missing.
        ValueError: If `mode` is invalid, or if `weights` are missing for weighted mode.
    """
    required = [columns_cost, columns_effectiveness]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    if mode not in {"standard", "balanced", "weighted"}:
        raise ValueError("mode must be one of {'standard', 'balanced', 'weighted'}")

    rng = np.random.default_rng(random_state)

    x = df[columns_cost].to_numpy(dtype=float)
    y = df[columns_effectiveness].to_numpy(dtype=float)

    # Row-wise sampling probabilities (for 'weighted' only)
    if mode == "weighted":
        if weights is None:
            raise ValueError("Please provide `weights` when using mode='weighted'.")
        if group_col not in df.columns:
            raise KeyError(f"Missing group_col '{group_col}' needed for weighted mode.")
        gw = df[group_col].map(lambda g: float(weights.get(g, 0.0))).to_numpy()
        if gw.sum() <= 0:
            raise ValueError("All provided weights are zero or missing.")
        p = gw / gw.sum()
    else:
        p = None  # uniform

    # Indices for balanced sampling
    if mode == "balanced":
        if group_col not in df.columns:
            raise KeyError(f"Missing group_col '{group_col}' needed for balanced mode.")
        groups = df[group_col].astype("category")
        codes = groups.cat.codes.to_numpy()
        # For each group, sample the same number of rows: min group size
        sizes = groups.value_counts().sort_index().to_numpy()
        gmin = int(sizes.min())
        if gmin <= 0:
            raise ValueError("At least one group is empty; cannot run balanced bootstrap.")
        # Precompute index arrays per group code
        group_indices: list[np.ndarray] = [np.flatnonzero(codes == gc) for gc in range(len(sizes))]

    boot_cost: list[float] = []
    boot_eff: list[float] = []

    n_rows = len(df)
    for _ in range(int(n)):
        if mode == "standard":
            idx = rng.integers(0, n_rows, size=n_rows, endpoint=False)
        elif mode == "weighted":
            idx = rng.choice(n_rows, size=n_rows, replace=True, p=p)
        else:  # balanced
            parts = [rng.choice(gidx, size=gmin, replace=True) for gidx in group_indices]
            idx = np.concatenate(parts, axis=0)

        boot_cost.append(float(np.mean(x[idx])))
        boot_eff.append(float(np.mean(y[idx])))

    out = pd.DataFrame({"delta_cost": boot_cost, "delta_effectiveness": boot_eff})
    if label is not None:
        out[group_col] = label
    return out

def plot_cost(
        ax,
        df,
        *,
        y_col: str = "cost_per_case",
        x_col: str = "experience_level",
        hue: str = "ai_diagnosis",
        exp_order: Sequence = (0, 1, 2),
        exp_labels: Sequence[str] = ("Junior", "Intermediate", "Senior"),
        ai_order: Sequence = (0, 1),
        x_label: str = "Experience Level",
        palette=None,
        ylim: Tuple[float, float] = (1500, 2400),
        ai_df=None, gt_df=None,
        baseline_ai: Optional[float] = None,
        baseline_gt: Optional[float] = None,
        baseline_agg = np.mean,
        swarm_size: float = 3,
        swarm_alpha: float = 0.5,
        fliersize: float = 0,
        show_legend: bool = False,
        despine: bool = True,
    ):
        """Plot cost per case vs. experience level with box+swarm plots."""
        if palette is None:
            palette = sns.color_palette("Greens", n_colors=len(ai_order))

        sns.boxplot(
            data=df, x=x_col, y=y_col,
            hue=hue, order=exp_order, hue_order=ai_order,
            palette=palette, fliersize=fliersize, ax=ax
        )
        sns.swarmplot(
            data=df, x=x_col, y=y_col,
            hue=hue, order=exp_order, hue_order=ai_order,
            dodge=True, color="black", alpha=swarm_alpha, size=swarm_size, ax=ax
        )

        if not show_legend:
            ax.legend_.remove() if ax.legend_ else None

        ax.set_ylim(*ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Expected Cost per Case [EUR]")
        ax.set_xticks(range(len(exp_order)))
        ax.set_xticklabels(exp_labels)

        ai_line = _maybe_baseline(baseline_ai, ai_df, y_col, baseline_agg)
        gt_line = _maybe_baseline(baseline_gt, gt_df, y_col, baseline_agg)
        _draw_baselines(ax, len(exp_order), ai_line, gt_line)

        if despine: sns.despine(ax=ax)
        return ax

def plot_effectiveness(
        ax,
        df,
        *,
        y_col: str = "effectiveness_per_case",
        x_col: str = "experience_level",
        hue: str = "ai_diagnosis",
        exp_order: Sequence = (0, 1, 2),
        exp_labels: Sequence[str] = ("Junior", "Intermediate", "Senior"),
        ai_order: Sequence = (0, 1),
        x_label: str = "Experience Level",
        palette=None,
        ylim: Tuple[float, float] = (0.75, 0.95),
        ai_df=None, gt_df=None,
        baseline_ai: Optional[float] = None,
        baseline_gt: Optional[float] = None,
        baseline_agg = np.mean,
        swarm_size: float = 3,
        swarm_alpha: float = 0.5,
        fliersize: float = 0,
        show_legend: bool = False,
        despine: bool = True,
    ):
        """Plot effectiveness vs. experience level with box+swarm plots."""
        if palette is None:
            palette = sns.color_palette("Blues", n_colors=len(ai_order))

        sns.boxplot(
            data=df, x=x_col, y=y_col,
            hue=hue, order=exp_order, hue_order=ai_order,
            palette=palette, fliersize=fliersize, ax=ax
        )
        sns.swarmplot(
            data=df, x=x_col, y=y_col,
            hue=hue, order=exp_order, hue_order=ai_order,
            dodge=True, color="black", alpha=swarm_alpha, size=swarm_size, ax=ax
        )

        if not show_legend:
            ax.legend_.remove() if ax.legend_ else None

        ax.set_ylim(*ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Expected Effectiveness per Case")
        ax.set_xticks(range(len(exp_order)))
        ax.set_xticklabels(exp_labels)

        ai_line = _maybe_baseline(baseline_ai, ai_df, y_col, baseline_agg)
        gt_line = _maybe_baseline(baseline_gt, gt_df, y_col, baseline_agg)
        _draw_baselines(ax, len(exp_order), ai_line, gt_line)

        if despine: sns.despine(ax=ax)
        return ax

def plot_ce_plane_jointgrid(
    df: pd.DataFrame,
    *,
    x: str = "delta_effectiveness",
    y: str = "delta_cost",
    hue: Optional[str] = "experience_level",
    hue_order: Optional[Sequence] = ("Junior", "Intermediate", "Senior"),
    palette: Union[str, Sequence, Mapping] = "viridis",
    height: float = 3.0,

    # scatter styling
    scatter_alpha: float = 0.12,
    scatter_size: float = 5.0,
    scatter_marker: str = "o",
    scatter_edgewidth: float = 0.0,   # avoids edgecolor warnings
    show_scatter_legend: bool = False,

    # KDE styling
    kde_fill: bool = True,
    kde_alpha: float = 0.4,
    kde_multiple: str = "layer",

    # confidence ellipse
    draw_ellipses: bool = True,
    ellipse_n_std: float = 1.95,
    ellipse_alpha: float = 0.7,
    ellipse_linewidth: float = 2.0,
    ellipse_fill: bool = False,
    ellipse_colors: Optional[Mapping] = None,  # e.g., {"Junior": "#440154", ...}

    # axes lines, limits, labels
    draw_origin: bool = True,
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-250, 250),
    xlabel: str = "Δ Effectiveness [%]",
    ylabel: str = "Δ Cost [EUR]",

    # misc
    despine: bool = False,
    savepath: Optional[str] = None,
):
    """
    Create a CE plane JointGrid with scatter in the joint axis, KDE marginals, and per-hue confidence ellipses.
    Returns the seaborn.JointGrid.
    """

    g = sns.JointGrid(data=df, x=x, y=y, height=height)

    if hue is not None:
        levels = list(hue_order) if hue_order is not None else list(pd.unique(df[hue]))
        if isinstance(palette, (list, tuple)):
            pal = list(palette)
        elif isinstance(palette, dict):
            pal = [palette.get(k) for k in levels]
        else:
            pal = sns.color_palette(palette, n_colors=len(levels))
        color_map = {lvl: pal[i] for i, lvl in enumerate(levels)}
    else:
        levels = None
        color_map = {}

    sns.scatterplot(
        data=df,
        x=x, y=y,
        hue=hue,
        hue_order=levels if levels is not None else None,
        palette=color_map if color_map else None,
        alpha=scatter_alpha,
        s=scatter_size,
        marker=scatter_marker,
        linewidth=scatter_edgewidth,
        ax=g.ax_joint,
        legend=show_scatter_legend,
        rasterized=False,
    )

    if draw_ellipses and hue is not None:
        for lvl in levels:
            sub = df[df[hue] == lvl]
            if len(sub) < 2:
                continue

            color = (ellipse_colors.get(lvl) if ellipse_colors is not None
                     else color_map.get(lvl, "black"))

            xvals = sub[x].to_numpy()
            yvals = sub[y].to_numpy()
            if np.allclose(xvals.var(), 0) and np.allclose(yvals.var(), 0):
                continue

            cov = np.cov(xvals, yvals)
            if not np.all(np.isfinite(cov)):
                continue

            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height_ = 2 * ellipse_n_std * np.sqrt(np.maximum(vals, 0))
            mean_x, mean_y = np.mean(xvals), np.mean(yvals)

            ell = Ellipse(
                (mean_x, mean_y),
                width=width,
                height=height_,
                angle=theta,
                facecolor=color if ellipse_fill else "none",
                edgecolor=color,
                linewidth=ellipse_linewidth,
                alpha=ellipse_alpha
            )
            g.ax_joint.add_patch(ell)

    if hue is not None:
        sns.kdeplot(
            data=df, x=x,
            hue=hue,
            hue_order=levels if levels is not None else None,
            palette=color_map if color_map else None,
            multiple=kde_multiple,
            ax=g.ax_marg_x,
            fill=kde_fill,
            alpha=kde_alpha,
            legend=False,
        )
        sns.kdeplot(
            data=df, y=y,
            hue=hue,
            hue_order=levels if levels is not None else None,
            palette=color_map if color_map else None,
            multiple=kde_multiple,
            ax=g.ax_marg_y,
            fill=kde_fill,
            alpha=kde_alpha,
            legend=False,
        )
    else:
        sns.kdeplot(data=df, x=x, ax=g.ax_marg_x, fill=kde_fill, alpha=kde_alpha, legend=False)
        sns.kdeplot(data=df, y=y, ax=g.ax_marg_y, fill=kde_fill, alpha=kde_alpha, legend=False)

    if draw_origin:
        g.ax_joint.axhline(0, color="black", linestyle="--", linewidth=1)
        g.ax_joint.axvline(0, color="black", linestyle="--", linewidth=1)

    g.ax_joint.set_xlim(*xlim)
    g.ax_joint.set_ylim(*ylim)
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)

    if despine:
        sns.despine(ax=g.ax_joint)
        sns.despine(ax=g.ax_marg_x, bottom=True, left=False)
        sns.despine(ax=g.ax_marg_y, left=True, bottom=False)

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")

    return g


def _span(n_cats: int) -> Tuple[float, float]:
    return (-0.5, n_cats - 0.5)

def _maybe_baseline(
    value: Optional[float],
    df,
    col: str,
    agg
) -> Optional[float]:
    if value is not None:
        return float(value)
    if df is None:
        return None
    arr = np.asarray(df[col])
    if arr.size == 0:
        return None
    return float(agg(arr))

def _draw_baselines(ax, n_cats: int, ai_line: Optional[float], gt_line: Optional[float]):
    x0, x1 = _span(n_cats)
    if ai_line is not None:
        ax.hlines(ai_line, x0, x1, colors="red", linestyles="dashed")
    if gt_line is not None:
        ax.hlines(gt_line, x0, x1, colors="black", linestyles="dashed")