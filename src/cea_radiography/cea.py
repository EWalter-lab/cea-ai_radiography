"""
Cost and effectiveness calculations for radiographic diagnostics.

This module standardizes and documents a small API to compute:
- Per-pathway treatment costs for TP/FP/FN/TN,
- Expected costs per tooth and per case (given prevalence and accuracy),
- Component cost breakdowns (absolute and percentage),
- Expected effectiveness per tooth and per case under simple 0/1 utilities.


Expected input columns
----------------------
- recall: float in [0, 1]
- specificity: float in [0, 1]
- ai_diagnosis: {0, 1} or probability in [0, 1] (AI used)
- TP_0.0, TP_1.0, TP_2.0: TP decision shares in percent (sum ≈ 100)
- FP_0.0, FP_1.0, FP_2.0: FP decision shares in percent (sum ≈ 100)

Added columns (costs)
---------------------
- tp_treatment_cost, fp_treatment_cost, fn_treatment_cost, tn_treatment_cost
- diagnostic_cost
- per_healthy_tooth_cost, per_diseased_tooth_cost
- cost_per_tooth, cost_per_case

Added columns (effectiveness)
-----------------------------
- tp_treatment_effectiveness, fp_treatment_effectiveness
- fn_treatment_effectiveness, tn_treatment_effectiveness
- effectiveness_per_positive, effectiveness_per_negative
- effectiveness_per_tooth, effectiveness_per_case

Pathway semantics (cost)
------------------------
- TP_0.0: examination + radiography + RCT + crown
- TP_1.0: CBCT + RCT + crown
- TP_2.0: RCT + crown
- FP_0.0: 0
- FP_1.0: radiography
- FP_2.0: RCT + crown
- FN: examination + CBCT + RCT + crown
- TN: 0

Effectiveness utilities (unitless)
----------------------------------
- TP: 0 for TP_0.0; 1 for TP_1.0 and TP_2.0
- FP: 1 for FP_0.0 and FP_1.0; 0 for FP_2.0
- FN: 0
- TN: 1

Conventions and notes
---------------------
- All monetary values are currency-agnostic.
- Decision shares are percentages (0–100); recall/specificity are in [0, 1].
- Diagnostic cost = examination + radiography + ai_diagnosis * ai.
- Per-tooth expected cost mixes positive/negative outcomes using
  `disease_prevalence`; per-case cost scales by `teeth_count` and adds the
  per-case `diagnostic_cost`.
- As modeled here, TP_0.0 and FN include examination/radiography while
  `diagnostic_cost` is also added per case; adjust formulas if you wish to
  avoid double counting diagnostics in your analysis.
- Style: Black (88 cols) and Ruff; PEP 257-compliant docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import pandas as pd


# ============================== Data model ============================== #
@dataclass(frozen=True, slots=True)
class Cost:
    """Unit costs for diagnostic/treatment items (currency-agnostic).

    Attributes:
        cbct: Cost of CBCT scan.
        ai: Cost of AI diagnosis (per case if used).
        rct: Cost of root canal treatment.
        radiography: Cost of radiography (2D).
        crown: Cost of crown placement.
        examination: Cost of clinical examination.
    """
    cbct: float
    panoramic: float
    intraoral_radiography: float
    ai: float
    rct: float
    crown: float
    examination: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    ai_usage_col: str = "ai_diagnosis"

    recall_col: str = "recall"
    spec_col: str = "specificity"

    tp_cols: Tuple[str, str, str] = ("TP_0.0", "TP_1.0", "TP_2.0")
    fp_cols: Tuple[str, str, str] = ("FP_0.0", "FP_1.0", "FP_2.0")

    disease_prevalence: float = 0.0848
    teeth_count: int = 25


class CostEffectivenessCalculator:
    """Compute costs & effectiveness from a DataFrame with recall/specificity
    and decision-share columns."""
    def __init__(self, 
                costs:Cost,
                config: Optional[ModelConfig] = None
                ) -> None:
        self.costs = costs
        self.config = config or ModelConfig()
        self._df: Optional[pd.DataFrame] = None
        self.result_cost: Optional[pd.DataFrame] = None
        self.result_effectiveness: Optional[pd.DataFrame] = None

    def _tp_td_cost(self, 
                df: pd.DataFrame) -> pd.Series:
        """Calculation of TP treatment costs
        
        Args:
            df: DataFrame with decision-share columns."""
        c0, c1, c2 = self.config.tp_cols
        _ensure(df, [c0, c1, c2], 0.0)
        _validate_shares(df, [c0, c1, c2])

        tp0 = self.costs.examination + self.costs.intraoral_radiography + self.costs.rct + self.costs.crown
        tp1 = self.costs.intraoral_radiography + self.costs.rct + self.costs.crown
        tp2 = self.costs.rct + self.costs.crown

        return (df[c0] / 100.0) * tp0 + (
            df[c1] / 100.0) * tp1 + (
            df[c2] / 100.0) * tp2

    def _fp_td_cost(self, 
                df: pd.DataFrame
                ) -> pd.Series:
        """Calculation of fp treatment costs
        
        Args:
            df: DataFrame with decision-share columns."""
        c0, c1, c2 = self.config.fp_cols
        _ensure(df, [c0, c1, c2], 0.0)
        _validate_shares(df, [c0, c1, c2])

        fp0 = 0
        fp1 = self.costs.intraoral_radiography
        fp2 = self.costs.rct + self.costs.crown

        return (df[c0] / 100.0) * fp0 + (
            df[c1] / 100.0) * fp1 + (
            df[c2] / 100.0) * fp2

    def _fn_td_cost(self) -> float:
        """Calculation of FN treatment costs"""
        return (self.costs.examination + 
            self.costs.intraoral_radiography + 
            self.costs.rct + 
            self.costs.crown)

    def _tn_td_cost(self) -> float:
        """Calculation of TN treatment costs"""
        return 0
    
    def _diagnostic_cost(self, 
                df: pd.DataFrame
                ) -> pd.Series:
        """Calculation of diagnostic costs
        
        Args:
            df: DataFrame with ai usage column.
        """
        ai_col = self.config.ai_usage_col
        _ensure(df, [ai_col], 0.0)
        _assert_01(df[ai_col], ai_col)

        base = self.costs.examination + self.costs.panoramic
        ai = self.costs.ai

        return base + df[ai_col] * ai
    
    def _per_healthy_tooth_cost(self, 
                df: pd.DataFrame
                ) -> pd.Series:
        """Per-healthy-tooth cost calculation
        
        Args:
            df: DataFrame with decision-share and ai usage columns.
        """
        _require(df, [self.config.spec_col])
        _assert_01(df[self.config.spec_col], self.config.spec_col)

        tn_cost = self._tn_td_cost()
        fp_cost = self._fp_td_cost(df)

        spec = df[self.config.spec_col]

        return (spec * tn_cost + (1 - spec) * fp_cost)

    def _per_diseased_tooth_cost(self, 
                df: pd.DataFrame
                ) -> pd.Series:
        """Per-diseased-tooth cost calculation
        
        Args:
            df: DataFrame with decision-share and ai usage columns.
        """
        _require(df, [self.config.recall_col])
        _assert_01(df[self.config.recall_col], self.config.recall_col)

        tp_cost = self._tp_td_cost(df)
        fn_cost = self._fn_td_cost()

        recall = df[self.config.recall_col]

        return (recall * tp_cost + (1 - recall) * fn_cost)
    
    def _per_tooth_cost(self, 
                df: pd.DataFrame
                ) -> pd.Series:
        """Per-tooth cost calculation
        
        Args:
            df: DataFrame with decision-share and ai usage columns.
        """
        prev = self.config.disease_prevalence

        diseased_cost = self._per_diseased_tooth_cost(df)
        healthy_cost = self._per_healthy_tooth_cost(df)

        return (prev * diseased_cost + (1 - prev) * healthy_cost)
    
    def _per_case_cost(self, 
                df: pd.DataFrame
                ) -> pd.Series:
        """Per-case cost calculation
        
        Args:
            df: DataFrame with decision-share and ai usage columns.
        """
        teeth = self.config.teeth_count
        return teeth * self._per_tooth_cost(df) + self._diagnostic_cost(df)
    
    def fit_cost(self,
            df: pd.DataFrame
            ) -> CostEffectivenessCalculator:
        """Fit the model to a DataFrame with recall/specificity and decision-share
        columns. The DataFrame is not modified in place; a copy with results is
        stored in the `result_` attribute.

        Args:
            df: DataFrame with recall/specificity and decision-share columns.

        Returns:
            The fitted calculator (self) with results in `result_` attribute.
        """
        out = df.copy()

        out["tp_treatment_cost"] = self._tp_td_cost(out)
        out["fp_treatment_cost"] = self._fp_td_cost(out)
        out["fn_treatment_cost"] = self._fn_td_cost()
        out["tn_treatment_cost"] = self._tn_td_cost()
        out["diagnostic_cost"] = self._diagnostic_cost(out)
        out["per_healthy_tooth_cost"] = self._per_healthy_tooth_cost(out)
        out["per_diseased_tooth_cost"] = self._per_diseased_tooth_cost(out)
        out["cost_per_tooth"] = self._per_tooth_cost(out)
        out["cost_per_case"] = self._per_case_cost(out)

        self.result_ = out
        return self
    
    def _tp_td_effectiveness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate TP treatment effectiveness from decision shares.

        Args:
            df: DataFrame with TP shares in percent, columns from `config.tp_cols`.

        Returns:
            Series with per-row TP effectiveness (unitless, 0..1).
        """
        c0, c1, c2 = self.config.tp_cols
        _ensure(df, [c0, c1, c2], 0.0)
        _validate_shares(df, [c0, c1, c2])

        return (df[c0] / 100.0) * 0.0 + (
            df[c1] / 100.0) * 1.0 + (
            df[c2] / 100.0) * 1.0

    def _fp_td_effectiveness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate FP treatment effectiveness from decision shares.

        Args:
            df: DataFrame with FP shares in percent, columns from `config.fp_cols`.

        Returns:
            Series with per-row FP effectiveness (unitless, 0..1).
        """
        c0, c1, c2 = self.config.fp_cols
        _ensure(df, [c0, c1, c2], 0.0)
        _validate_shares(df, [c0, c1, c2])

        return (df[c0] / 100.0) * 1.0 + (
            df[c1] / 100.0) * 1.0 + (
            df[c2] / 100.0) * 0.0

    def _fn_td_effectiveness(self) -> float:
        """Calculate FN treatment effectiveness.

        Returns:
            Scalar FN effectiveness (unitless).
        """
        return 0.0

    def _tn_td_effectiveness(self) -> float:
        """Calculate TN treatment effectiveness.

        Returns:
            Scalar TN effectiveness (unitless).
        """
        return 1.0

    def _per_positive_tooth_effectiveness(self, df: pd.DataFrame) -> pd.Series:
        """Per-positive-tooth effectiveness (given disease is present).

        Formula:
            E_pos = recall * TP_eff + (1 - recall) * FN_eff

        Args:
            df: DataFrame with recall column as in `config.recall_col`.

        Returns:
            Series with expected effectiveness per diseased tooth (unitless).
        """
        _require(df, [self.config.recall_col])
        _assert_01(df[self.config.recall_col], self.config.recall_col)

        tp_eff = self._tp_td_effectiveness(df)
        fn_eff = self._fn_td_effectiveness()
        r = df[self.config.recall_col]

        return r * tp_eff + (1.0 - r) * fn_eff

    def _per_negative_tooth_effectiveness(self, df: pd.DataFrame) -> pd.Series:
        """Per-negative-tooth effectiveness (given disease is absent).

        Formula:
            E_neg = specificity * TN_eff + (1 - specificity) * FP_eff

        Args:
            df: DataFrame with specificity column as in `config.spec_col`.

        Returns:
            Series with expected effectiveness per healthy tooth (unitless).
        """
        _require(df, [self.config.spec_col])
        _assert_01(df[self.config.spec_col], self.config.spec_col)

        fp_eff = self._fp_td_effectiveness(df)
        tn_eff = self._tn_td_effectiveness()
        s = df[self.config.spec_col]

        return s * tn_eff + (1.0 - s) * fp_eff

    def _per_tooth_effectiveness(self, df: pd.DataFrame) -> pd.Series:
        """Per-tooth effectiveness, mixing positive/negative by prevalence.

        Formula:
            E_tooth = p * E_pos + (1 - p) * E_neg

        Args:
            df: DataFrame with recall/specificity and decision-share columns.

        Returns:
            Series with expected effectiveness per tooth (unitless).
        """
        p = self.config.disease_prevalence
        e_pos = self._per_positive_tooth_effectiveness(df)
        e_neg = self._per_negative_tooth_effectiveness(df)
        return p * e_pos + (1.0 - p) * e_neg

    def _per_case_effectiveness(self, df: pd.DataFrame) -> pd.Series:
        """Per-case effectiveness (unitless).

        Notes:
            Unlike costs, effectiveness is typically not scaled by `teeth_count`.
            If you explicitly want a tooth-scaled metric, multiply the per-tooth
            effectiveness by `teeth_count` outside this method.
        """
        return self._per_tooth_effectiveness(df)

    def fit_effectiveness(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute and append effectiveness columns to a DataFrame.

        Adds:
            tp_treatment_effectiveness, fp_treatment_effectiveness,
            fn_treatment_effectiveness, tn_treatment_effectiveness,
            effectiveness_per_positive, effectiveness_per_negative,
            effectiveness_per_tooth, effectiveness_per_case

        Args:
            df: Optional DataFrame. If omitted, uses `self.result_` from `fit(...)`.

        Returns:
            Copy of the input DataFrame with added effectiveness columns.
        """
        base = df if df is not None else self.result_
        if base is None:
            raise ValueError("Provide a DataFrame or call .fit(df) first.")
        out = base.copy()

        out["tp_treatment_effectiveness"] = self._tp_td_effectiveness(out)
        out["fp_treatment_effectiveness"] = self._fp_td_effectiveness(out)
        out["fn_treatment_effectiveness"] = self._fn_td_effectiveness()
        out["tn_treatment_effectiveness"] = self._tn_td_effectiveness()

        out["effectiveness_per_positive"] = self._per_positive_tooth_effectiveness(out)
        out["effectiveness_per_negative"] = self._per_negative_tooth_effectiveness(out)
        out["effectiveness_per_tooth"] = self._per_tooth_effectiveness(out)
        out["effectiveness_per_case"] = self._per_case_effectiveness(out)

        self.result_= out
        return out
    

# ============================ Helper functions =========================== #
def _ensure(df: pd.DataFrame, 
            cols: Iterable[str], 
            default: float = 0.0
            ) -> None:
    """Ensure columns exist; create with default if missing (in place)."""
    for c in cols:
        if c not in df.columns:
            df[c] = default

def _require(df: pd.DataFrame, 
            cols: Iterable[str]
            ) -> None:
    """Ensure columns exist; raise KeyError if missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

def _assert_01(s: pd.Series, 
            name: str) -> None:
    if ((s < 0) | (s > 1)).any():
        bad = s[(s < 0) | (s > 1)].head().tolist()
        raise ValueError(f"Column '{name}' must be within [0, 1]. Examples: {bad}")

def _assert_0_100(s: pd.Series, name: str) -> None:
    if ((s < 0) | (s > 100)).any():
        bad = s[(s < 0) | (s > 100)].head().tolist()
        raise ValueError(f"Column '{name}' must be within [0, 100]. Examples: {bad}")

def _validate_shares(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Validate that share columns sum to 100% (or 0 if all zero)."""
    total = df[list(cols)].sum(axis=1)
    all_zero = (total == 0).all()
    if not all_zero:
        bad = total[(total < 99.9) | (total > 100.1)].head().tolist()
        if bad:
            raise ValueError(f"Columns {cols} must sum to 100%. Examples: {bad}")