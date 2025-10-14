"""Performance metrics and FP/TP treatment-decision analysis.

This module provides:

- `PerformanceConfig`: Column names and evaluation behavior.
- `PerformanceEvaluator`: Vectorized computation of confusion counts/metrics
  (accuracy, precision, recall, specificity, F1, prevalence) and—optionally—
  FP/TP treatment-decision (%) distributions. Results can be grouped and can
  carry through metadata via configurable aggregations (e.g., mode).

Quick start
-----------
Compute metrics grouped by examiner and include treatment-decision columns:

>>> from cea_radiography import performance
>>> import pandas as pd
>>>
>>> filepath = "../01_Data/1_Data/master_dataset.csv"
>>> df = pd.read_csv(filepath)
>>>
>>> eval_cfg = performance.PerformanceConfig(
...     ground_truth_col="ground_truth",
...     diagnosis_col="diagnosis",
...     treatment_col="treatments",
...     examiner_col="examiner",
...     ai_diagnosis_col="ai_diagnosis",
...     groupby_cols=("examiner",),
...     passthrough_cols=("experience_level",),
...     passthrough_reduce="mode",
...     positive_label=1,
...     negative_label=0,
...     zero_division_value=0.0,
...     enforce_binary=True,
... )
>>> evaluator = performance.PerformanceEvaluator(eval_cfg)
>>> performance_df = evaluator.evaluate(df)
>>> performance_df.head()

Output columns (typical)
------------------------
- Group keys (e.g., `examiner`) and passthrough (e.g., `experience_level`)
- Metrics: `prevalence`, `tn`, `fp`, `fn`, `tp`, `accuracy`, `precision`,
  `recall`, `specificity`, `f1_score`, `confusion_matrix`
- FP/TP treatment shares as percentages: columns like `FP_0`, `FP_1`, ..., `TP_0`, `TP_1`, ...

Notes
-----
- Disable treatment decisions if desired:
    `evaluator.evaluate(df, include_treatmentdecision=False)`
- Change grouping at call time (overrides config):
    `evaluator.evaluate(df, by=("examiner", "ai_diagnosis"))`
- Passthrough aggregation options: `"mode"`, `"first"`, or a callable.
- Binary validation can be turned off with `enforce_binary=False`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================== Config ============================== #


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    """Column configuration and metric behavior (binary-only).

    Attributes:
        ground_truth_col: Column with CBCT gold label (0/1).
        diagnosis_col: Column with examiner decision (0/1).
        treatment_col: Column with treatment decision/category.
        examiner_col: Column identifying the examiner.
        ai_diagnosis_col: Column with AI suggestion/diagnosis flag (0/1).
        patient_id_col: Column identifying the patient/case.

        groupby_cols: Default grouping for `evaluate` (empty -> overall).
        passthrough_cols: Columns to pass through per group (e.g., "experience_level").
        passthrough_reduce: Aggregator for passthrough columns. One of:
            - "mode" (default),
            - "first",
            - callable: a function(pd.Series) -> scalar.
        positive_label: Positive class value (default 1).
        negative_label: Negative class value (default 0).
        zero_division_value: Fallback when denominator is zero (default 0.0).
        enforce_binary: Validate that inputs are in {0,1}.
    """

    ground_truth_col: str = "ground_truth"
    diagnosis_col: str = "diagnosis"
    treatment_col: str = "treatments"
    examiner_col: str = "examiner"
    ai_diagnosis_col: str = "ai_diagnosis"
    patient_id_col: str = "patient"

    groupby_cols: Tuple[str, ...] = ()
    passthrough_cols: Tuple[str, ...] = ("experience_level",)
    passthrough_reduce: str | Callable[[pd.Series], object] = "mode"
    positive_label: int = 1
    negative_label: int = 0
    zero_division_value: float = 0.0
    enforce_binary: bool = True


# ============================== Evaluator ============================== #


class PerformanceEvaluator:
    """Compute confusion counts/metrics vs. CBCT ground truth; keep selected metadata."""

    _METRIC_ORDER: Tuple[str, ...] = (
        "prevalence",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "confusion_matrix",
    )

    def __init__(self, config: PerformanceConfig = PerformanceConfig()) -> None:
        """Initialize the evaluator.

        Args:
            config: Configuration with column names and behavior flags.
        """
        self.config = config

    # -------- Public API -------- #
    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        by: Sequence[str] | None = None,
        passthrough: Sequence[str] | None = None,
        passthrough_reduce: str | Callable[[pd.Series], object] | None = None,
        include_treatmentdecision: bool = True,
    ) -> pd.DataFrame:
        """Compute counts/metrics and (optionally) FP/TP treatment distributions.

        This orchestrates **independent** computations from a shared prepared table:
        1) Prepare once (`_prepare_base`).
        2) Counts/metrics from the prepared table.
        3) Treatment decision (FP_/TP_) from the prepared table.
        4) Merge results at the end.

        Treatment decision columns are merged at the minimal meaningful grain:
        - If `by` includes any of {`ai_diagnosis`, `examiner`}, TD is computed at those
          keys and merged (avoids duplication).
        - If `by` is empty (overall), overall TD columns are appended as one row.
        - Otherwise (grouped by unrelated keys), TD is skipped to avoid misleading joins.

        Args:
            df: Input DataFrame.
            by: Grouping columns (defaults to `config.groupby_cols`).
            passthrough: Columns carried through per group
                (defaults to `config.passthrough_cols`).
            passthrough_reduce: Aggregation for passthrough columns
                ("mode" | "first" | callable). Defaults to `config.passthrough_reduce`.
            include_treatmentdecision: Whether to attach FP_/TP_ treatment percentage
                columns (default True).

        Returns:
            A DataFrame with grouping keys (optional), passthrough columns, counts,
            metrics, and possibly FP_/TP_ treatment percentage columns.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If binary columns contain values outside {0,1} and
                `config.enforce_binary` is True.
        """
        cfg = self.config
        ai_col = cfg.ai_diagnosis_col
        examiner_col = cfg.examiner_col
        exp_default = "experience_level"

        by = list(by) if by else list(cfg.groupby_cols)

        # ---------- 1) Prepare once ----------
        prepared = self._prepare_base(df)

        # ---------- 2) Counts/metrics (independent) ----------
        counts = self._counts_from_prepared(prepared, by=by)
        out = self._add_metrics(counts)

        # ---------- 3) Passthrough (independent) ----------
        passthrough = (
            list(passthrough)
            if passthrough is not None
            else list(cfg.passthrough_cols)
        )
        if passthrough:
            reduce_fn = (
                passthrough_reduce
                if passthrough_reduce is not None
                else cfg.passthrough_reduce
            )
            meta = self._aggregate_passthrough(
                df,  # passthrough may include columns not in prepared
                by=by,
                cols=passthrough,
                reducer=reduce_fn,
            )
            out = meta.merge(out, on=by, how="right") if by else pd.concat(
                [meta.reset_index(drop=True), out], axis=1
            )

        # ---------- 4) Treatment decisions ----------
        if include_treatmentdecision:
            merge_keys = [k for k in by if k in {ai_col, examiner_col}]
            if merge_keys:
                td_df = self._treatmentdecision_from_prepared(
                    prepared=prepared,
                    index_keys=merge_keys,
                    exp_col=exp_default,
                )
                if (
                    examiner_col in merge_keys
                    and exp_default in out.columns
                    and exp_default in td_df.columns
                ):
                    merge_keys = [*merge_keys, exp_default]
                out = out.merge(td_df, on=merge_keys, how="left")
            elif not by:
                td_overall = self._treatmentdecision_from_prepared(
                    prepared=prepared,
                    index_keys=[],
                    exp_col="__none__",
                )
                if not td_overall.empty:
                    out = pd.concat(
                        [out.reset_index(drop=True), td_overall.reset_index(drop=True)],
                        axis=1,
                    )

        # ---------- Final column order ----------
        fp_cols = sorted(
            [c for c in out.columns if c.startswith("FP_")],
            key=self._td_sort_key,
        )
        tp_cols = sorted(
            [c for c in out.columns if c.startswith("TP_")],
            key=self._td_sort_key,
        )

        leading = (by if by else []) + (passthrough if passthrough else [])
        metrics_present = [c for c in self._METRIC_ORDER if c in out.columns]
        if "confusion_matrix" in metrics_present:
            metrics_present.remove("confusion_matrix")
            ordered = (
                leading
                + metrics_present
                + fp_cols
                + tp_cols
                + ["confusion_matrix"]
            )
        else:
            ordered = leading + metrics_present + fp_cols + tp_cols

        ordered = [c for c in ordered if c in out.columns]
        ordered += [c for c in out.columns if c not in ordered]
        return out[ordered]

    # -------- Main building blocks -------- #
    def _prepare_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a reusable table with validated binary labels and helper columns.

        Builds typed/validated ground-truth and diagnosis vectors, then computes:
        - `_tn`, `_fp`, `_fn`, `_tp` indicator columns (0/1).
        - `_case_type` in {"FP","TP", None} for rows with diagnosed-positive.

        Uses an object-typed container for `_case_type` to avoid NumPy's dtype
        promotion issues when mixing strings with missing values.

        Args:
            df: Input DataFrame.

        Returns:
            A copy of `df` with `_tn`, `_fp`, `_fn`, `_tp`, and `_case_type` added.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If labels are not binary and `config.enforce_binary` is True.
        """
        cfg = self.config
        self._require(df, [cfg.ground_truth_col, cfg.diagnosis_col])

        base = df.copy()
        gt = (
            pd.to_numeric(base[cfg.ground_truth_col], errors="coerce")
            .fillna(cfg.negative_label)
            .astype(int)
        )
        pr = (
            pd.to_numeric(base[cfg.diagnosis_col], errors="coerce")
            .fillna(cfg.negative_label)
            .astype(int)
        )

        if cfg.enforce_binary:
            self._assert_binary(gt, cfg.ground_truth_col)
            self._assert_binary(pr, cfg.diagnosis_col)

        base["_tn"] = ((gt == cfg.negative_label) & (pr == cfg.negative_label)).astype(int)
        base["_fp"] = ((gt == cfg.negative_label) & (pr == cfg.positive_label)).astype(int)
        base["_fn"] = ((gt == cfg.positive_label) & (pr == cfg.negative_label)).astype(int)
        base["_tp"] = ((gt == cfg.positive_label) & (pr == cfg.positive_label)).astype(int)

        case_type = np.full(len(base), None, dtype=object)
        pos = pr == cfg.positive_label
        case_type[pos] = np.where(gt[pos] == cfg.negative_label, "FP", "TP")
        base["_case_type"] = case_type

        return base

    def _counts_from_prepared(
        self,
        prepared: pd.DataFrame,
        *,
        by: Sequence[str],
    ) -> pd.DataFrame:
        """Compute tn/fp/fn/tp counts from a prepared table.

        Args:
            prepared: Output of `_prepare_base`.
            by: Grouping columns (empty -> overall).

        Returns:
            Counts DataFrame with tn, fp, fn, tp (+ group keys if `by` specified).
        """
        need = ["_tn", "_fp", "_fn", "_tp"] + (list(by) if by else [])
        self._require(prepared, need)

        if by:
            counts = (
                prepared.groupby(list(by), dropna=False)[["_tn", "_fp", "_fn", "_tp"]]
                .sum()
                .rename(columns={"_tn": "tn", "_fp": "fp", "_fn": "fn", "_tp": "tp"})
                .reset_index()
            )
        else:
            counts = pd.DataFrame(
                {
                    "tn": [int(prepared["_tn"].sum())],
                    "fp": [int(prepared["_fp"].sum())],
                    "fn": [int(prepared["_fn"].sum())],
                    "tp": [int(prepared["_tp"].sum())],
                }
            )
        return counts

    def _add_metrics(self, counts: pd.DataFrame) -> pd.DataFrame:
        """Add vectorized metrics to a counts DataFrame.

        Args:
            counts: DataFrame with columns tn, fp, fn, tp.

        Returns:
            DataFrame with added metric columns:
            accuracy, precision, recall, specificity, f1_score, prevalence,
            and confusion_matrix (2x2 as nested lists).
        """
        zv = self.config.zero_division_value
        out = counts.copy()

        tn = out["tn"].astype(float)
        fp = out["fp"].astype(float)
        fn = out["fn"].astype(float)
        tp = out["tp"].astype(float)

        out["accuracy"] = self._metric_accuracy(tn, fp, fn, tp)
        out["precision"] = self._metric_precision(tp, fp, zero_value=zv)
        out["recall"] = self._metric_recall(tp, fn, zero_value=zv)
        out["specificity"] = self._metric_specificity(tn, fp, zero_value=zv)
        out["f1_score"] = self._metric_f1(out["precision"], out["recall"])
        out["prevalence"] = self._prevalence(tn, fp, fn, tp)

        out["confusion_matrix"] = out[["tn", "fp", "fn", "tp"]].apply(
            lambda r: [[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]],
            axis=1,
        )
        return out

    def _treatmentdecision_from_prepared(
        self,
        *,
        prepared: pd.DataFrame,
        index_keys: Sequence[str],
        exp_col: str = "experience_level",
    ) -> pd.DataFrame:
        """Compute FP/TP treatment percentages from a prepared table.

        Args:
            prepared: Output of `_prepare_base`.
            index_keys: Keys for grouping TD shares (e.g., ["ai_diagnosis", "examiner"]);
                empty list returns a single-row overall table.
            exp_col: Optional experience column; merged (mode per examiner) when
                "examiner" in `index_keys` and `exp_col` exists in `prepared`.

        Returns:
            A wide DataFrame with FP_<treatment>, TP_<treatment> columns and
            optional `exp_col` passthrough.
        """
        cfg = self.config
        treat_col = cfg.treatment_col
        self._require(prepared, [treat_col] + list(index_keys))

        base = prepared.loc[pd.notna(prepared["_case_type"])].copy()
        base = base.dropna(subset=[treat_col])
        if base.empty:
            return pd.DataFrame(columns=list(index_keys))

        if not index_keys:
            counts = (
                base.groupby(["_case_type", treat_col], dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            totals = (
                base.groupby(["_case_type"], dropna=False)
                .size()
                .rename("total")
                .reset_index()
            )
            stats = counts.merge(totals, on="_case_type", how="left")
            stats["percent"] = stats["count"] / stats["total"] * 100.0

            folded: dict[str, float] = {}
            for _, r in stats.iterrows():
                folded[f"{r['_case_type']}_{r[treat_col]}"] = float(r["percent"])
            return pd.DataFrame([folded])

        counts = (
            base[[*index_keys, "_case_type", treat_col]]
            .value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
        totals = (
            base[[*index_keys, "_case_type"]]
            .value_counts(dropna=False)
            .rename("total")
            .reset_index()
        )
        stats = counts.merge(totals, on=[*index_keys, "_case_type"], how="left")
        stats["percent"] = stats["count"] / stats["total"] * 100.0

        stats["col"] = stats["_case_type"] + "_" + stats[treat_col].astype(str)
        wide = (
            stats.pivot_table(
                index=list(index_keys),
                columns="col",
                values="percent",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )

        examiner_col = cfg.examiner_col
        if exp_col in prepared.columns and examiner_col in index_keys:
            exp_map = (
                prepared[[examiner_col, exp_col]]
                .dropna(subset=[examiner_col])
                .groupby(examiner_col, dropna=False)[exp_col]
                .agg(self._mode_scalar)
                .reset_index()
            )
            wide = wide.merge(exp_map, on=examiner_col, how="left")

        id_cols = list(index_keys) + ([exp_col] if exp_col in wide.columns else [])
        fp_cols = sorted([c for c in wide.columns if c.startswith("FP_")], key=self._td_sort_key)
        tp_cols = sorted([c for c in wide.columns if c.startswith("TP_")], key=self._td_sort_key)
        other_cols = [c for c in wide.columns if c not in (id_cols + fp_cols + tp_cols)]
        return wide[id_cols + fp_cols + tp_cols + other_cols]

    # -------- Backward-compatible thin wrappers (optional) -------- #
    def _evaluate_treatmentdecision_FP_TP(
        self,
        df: pd.DataFrame,
        *,
        exp_col: str = "experience_level",
        print_results: bool = False,
    ) -> pd.DataFrame:
        """Backward-compatible wrapper for per-(ai_diagnosis × examiner) TD.

        Args:
            df: Input DataFrame.
            exp_col: Experience column to add as passthrough (mode per examiner), if
                present in `df`. Default is "experience_level".
            print_results: If True, pretty-prints the wide table (debug only).

        Returns:
            Wide TD table at (ai_diagnosis, examiner) grain.
        """
        prepared = self._prepare_base(df)
        out = self._treatmentdecision_from_prepared(
            prepared=prepared,
            index_keys=[self.config.ai_diagnosis_col, self.config.examiner_col],
            exp_col=exp_col,
        )
        if print_results and not out.empty:
            with pd.option_context("display.max_columns", None, "display.width", 180):
                print(out.to_string(index=False))
        return out

    def _evaluate_treatmentdecision_overall(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible wrapper for overall TD (single row).

        Args:
            df: Input DataFrame.

        Returns:
            Single-row DataFrame with FP_<treatment> and TP_<treatment>.
        """
        prepared = self._prepare_base(df)
        return self._treatmentdecision_from_prepared(
            prepared=prepared,
            index_keys=[],
            exp_col="__none__",
        )

    # -------- Passthrough aggregation -------- #
    def _aggregate_passthrough(
        self,
        df: pd.DataFrame,
        *,
        by: Sequence[str],
        cols: Sequence[str],
        reducer: str | Callable[[pd.Series], object] = "mode",
    ) -> pd.DataFrame:
        """Aggregate `cols` per group using `reducer`.

        Args:
            df: Input DataFrame.
            by: Grouping columns; if empty, returns a single-row aggregation.
            cols: Column names to aggregate.
            reducer: Aggregation strategy:
                - "mode": First mode (ties broken by position).
                - "first": First non-null value.
                - callable: Function mapping a Series to a scalar.

        Returns:
            DataFrame of aggregated passthrough columns per group (or single row).

        Raises:
            ValueError: If `reducer` is an unsupported string.
        """
        for c in cols:
            if c not in df.columns:
                df = df.assign(**{c: np.nan})

        if callable(reducer):
            aggfunc = reducer
        elif reducer == "mode":
            aggfunc = self._mode_scalar
        elif reducer == "first":
            aggfunc = self._first_scalar
        else:
            raise ValueError("Unsupported passthrough_reduce. Use 'mode', 'first', or a callable.")

        if by:
            meta = (
                df.groupby(list(by), dropna=False)[list(cols)]
                .agg(aggfunc)
                .reset_index()
            )
        else:
            meta = pd.DataFrame([{c: aggfunc(df[c]) for c in cols}])
        return meta

    @staticmethod
    def _mode_scalar(s: pd.Series) -> object:
        """Return the first mode in a Series (ignoring NA).

        Args:
            s: Input Series.

        Returns:
            Mode value or NaN if the Series is empty after dropping NA.
        """
        s = s.dropna()
        if s.empty:
            return np.nan
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else np.nan

    @staticmethod
    def _first_scalar(s: pd.Series) -> object:
        """Return the first non-null element of a Series.

        Args:
            s: Input Series.

        Returns:
            First non-null element or NaN if none exist.
        """
        s = s.dropna()
        return s.iloc[0] if not s.empty else np.nan

    # -------- Metric helpers (vectorized) -------- #
    @staticmethod
    def _metric_accuracy(
        tn: pd.Series | np.ndarray,
        fp: pd.Series | np.ndarray,
        fn: pd.Series | np.ndarray,
        tp: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Compute accuracy = (tp + tn) / (tn + fp + fn + tp)."""
        total = tn + fp + fn + tp
        return np.where(total > 0, (tp + tn) / total, np.nan)

    @staticmethod
    def _metric_precision(
        tp: pd.Series | np.ndarray,
        fp: pd.Series | np.ndarray,
        *,
        zero_value: float = 0.0,
    ) -> np.ndarray:
        """Compute precision = tp / (tp + fp).

        Args:
            tp: True positive counts.
            fp: False positive counts.
            zero_value: Value used when denominator is zero.

        Returns:
            Precision values.
        """
        denom = tp + fp
        return np.where(denom > 0, tp / denom, zero_value)

    @staticmethod
    def _metric_recall(
        tp: pd.Series | np.ndarray,
        fn: pd.Series | np.ndarray,
        *,
        zero_value: float = 0.0,
    ) -> np.ndarray:
        """Compute recall (sensitivity) = tp / (tp + fn).

        Args:
            tp: True positive counts.
            fn: False negative counts.
            zero_value: Value used when denominator is zero.

        Returns:
            Recall values.
        """
        denom = tp + fn
        return np.where(denom > 0, tp / denom, zero_value)

    @staticmethod
    def _metric_specificity(
        tn: pd.Series | np.ndarray,
        fp: pd.Series | np.ndarray,
        *,
        zero_value: float = 0.0,
    ) -> np.ndarray:
        """Compute specificity = tn / (tn + fp).

        Args:
            tn: True negative counts.
            fp: False positive counts.
            zero_value: Value used when denominator is zero.

        Returns:
            Specificity values.
        """
        denom = tn + fp
        return np.where(denom > 0, tn / denom, zero_value)

    @staticmethod
    def _metric_f1(
        precision: pd.Series | np.ndarray,
        recall: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Compute F1 score = 2 * precision * recall / (precision + recall)."""
        denom = precision + recall
        return np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    @staticmethod
    def _prevalence(
        tn: pd.Series | np.ndarray,
        fp: pd.Series | np.ndarray,
        fn: pd.Series | np.ndarray,
        tp: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Compute prevalence = (tp + fn) / (tn + fp + fn + tp)."""
        total = tn + fp + fn + tp
        return np.where(total > 0, (tp + fn) / total, np.nan)

    # -------- validators & small utilities -------- #
    @staticmethod
    def _require(df: pd.DataFrame, cols: Iterable[str]) -> None:
        """Validate that required columns exist.

        Args:
            df: Input DataFrame.
            cols: Column names required in `df`.

        Raises:
            KeyError: If any of the required columns are missing.
        """
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")

    @staticmethod
    def _assert_binary(s: pd.Series, name: str) -> None:
        """Assert a Series contains only {0,1} after numeric coercion.

        Args:
            s: Series to validate.
            name: Column name (for error messages).

        Raises:
            ValueError: If values outside {0,1} are present.
        """
        vals = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique())
        if not vals.issubset({0, 1}):
            raise ValueError(
                f"Column '{name}' must be binary {{0,1}}; seen: {sorted(vals)}"
            )

    @staticmethod
    def _td_sort_key(c: str) -> tuple[int, str, float | str]:
        """Sort key for FP_/TP_ treatment columns: FP_* first, numeric-aware.

        Args:
            c: Column name like 'FP_0' or 'TP_2.0'.

        Returns:
            Sorting tuple used by `sorted`.
        """
        if "_" not in c:
            return (2, c, 0.0)
        prefix, rest = c.split("_", 1)
        order = 0 if prefix == "FP" else (1 if prefix == "TP" else 2)
        try:
            return (order, prefix, float(rest))
        except Exception:
            return (order, prefix, rest)
