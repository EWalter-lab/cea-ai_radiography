# ce-radiography

Cost and effectiveness calculations for radiographic diagnostics (paper code).

This package provides a small, reproducible API to compute:

- Per-pathway treatment costs for TP/FP/FN/TN  
- Expected costs per tooth and per case (given prevalence & accuracy)  
- Component cost breakdowns (absolute and % of total)  
- Expected effectiveness per tooth and per case under simple 0/1 utilities

It’s designed to be transparent and easy to test.

---

## Contents

- [Installation](#installation)  
- [Quickstart](#quickstart)  
- [Expected input columns](#expected-input-columns)  
- [Pathway semantics](#pathway-semantics)  
- [API overview](#api-overview)  
- [Reproducibility & modeling notes](#reproducibility--modeling-notes)  
- [Project layout](#project-layout)  
- [Development (Black/Ruff/pytest)](#development-blackruffpytest)  
- [Cite](#cite)  
- [License](#license)

---

## Installation

```bash
# clone your repo
git clone <your-repo-url> && cd <repo>

# optional: create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install package (editable) + dev tools
pip install -e .
pip install black ruff pytest
```

If you’re using the provided `pyproject.toml`, you can also do:

```bash
pip install -e ".[dev]"
```

---

## Quickstart

```python
import pandas as pd
from ce_radiography.model import Cost, ModelConfig, CostEffectivenessCalculator

# 1) Define unit costs (currency-agnostic)
costs = Cost(
    cbct=100.0,
    ai=5.0,
    rct=500.0,
    radiography=40.0,    # or 'intraoral_radiography' if you renamed the field
    crown=700.0,
    examination=60.0,
)

# 2) Create a DataFrame with model performance + decision shares (percent)
df = pd.DataFrame(
    {
        "recall": [0.8, 0.9],
        "specificity": [0.95, 0.90],
        "ai_diagnosis": [1, 0],     # 0/1 or probability in [0,1]
        "TP_0.0": [50, 0],
        "TP_1.0": [25, 50],
        "TP_2.0": [25, 50],
        "FP_0.0": [90, 80],
        "FP_1.0": [10, 10],         # in current code, FP_1.0 = radiography
        "FP_2.0": [0, 10],
    }
)

# 3) Configure epidemiology / column names if needed
cfg = ModelConfig(
    disease_prevalence=0.0848,  # per-tooth prevalence
    teeth_count=25,             # per-case tooth count (scaling factor)
)

# 4) Compute costs per tooth & per case
calc = CostEffectivenessCalculator(costs, cfg)
calc.fit(df)                 # stores results internally
costs_df = calc.result_      # latest result (sklearn-style convention)
print(costs_df.filter(regex="cost").round(2))

# 5) Append effectiveness columns (0/1 utilities)
eff_df = calc.add_effectiveness()  # uses previous result by default
print(eff_df.filter(regex="effect").round(3))
```

> Prefer stateless usage? You can adapt methods to accept a DataFrame and return a new DataFrame without using `result_`. Both styles are fine; pick one and keep it consistent.

---

## Expected input columns

Your DataFrame should include:

- `recall`: float in \[0, 1]  
- `specificity`: float in \[0, 1]  
- `ai_diagnosis`: {0,1} or probability in \[0,1] (AI used)  
- Decision shares (percent; rows usually sum to 100):  
  - `TP_0.0`, `TP_1.0`, `TP_2.0`  
  - `FP_0.0`, `FP_1.0`, `FP_2.0`  

Missing decision-share columns are created with 0.

---

## Pathway semantics

### Cost (current defaults)

- **TP_0.0**: examination + radiography + RCT + crown  
- **TP_1.0**: CBCT + RCT + crown  
- **TP_2.0**: RCT + crown  
- **FP_0.0**: 0  
- **FP_1.0**: radiography *(if you intend CBCT here, change it in the code)*  
- **FP_2.0**: RCT + crown  
- **FN**: examination + CBCT + RCT + crown  
- **TN**: 0  

### Effectiveness (unitless 0/1 utilities)

- **TP**: 0 for TP_0.0; 1 for TP_1.0 and TP_2.0  
- **FP**: 1 for FP_0.0 and FP_1.0; 0 for FP_2.0  
- **FN**: 0  
- **TN**: 1  

---

## API overview

### Dataclasses

```python
@dataclass(frozen=True, slots=True)
class Cost:
    cbct: float
    ai: float
    rct: float
    radiography: float  # rename consistently if you use 'intraoral_radiography'
    crown: float
    examination: float

@dataclass(frozen=True, slots=True)
class ModelConfig:
    ai_usage_col: str = "ai_diagnosis"
    recall_col: str = "recall"
    spec_col: str = "specificity"
    tp_cols: tuple[str, str, str] = ("TP_0.0", "TP_1.0", "TP_2.0")
    fp_cols: tuple[str, str, str] = ("FP_0.0", "FP_1.0", "FP_2.0")
    disease_prevalence: float = 0.0848
    teeth_count: int = 25
```

### Calculator (stateful, sklearn-style cache)

- `fit(df)` → stores a copy with these added columns:
  - `tp_treatment_cost`, `fp_treatment_cost`, `fn_treatment_cost`, `tn_treatment_cost`  
  - `diagnostic_cost`  
  - `per_healthy_tooth_cost`, `per_diseased_tooth_cost`  
  - `cost_per_tooth`, `cost_per_case`  
- `result_` → the most recent DataFrame computed by the calculator  
- `add_effectiveness(df: Optional[pd.DataFrame] = None)` → appends:
  - `tp_treatment_effectiveness`, `fp_treatment_effectiveness`,  
    `fn_treatment_effectiveness`, `tn_treatment_effectiveness`,  
    `effectiveness_per_positive`, `effectiveness_per_negative`,  
    `effectiveness_per_tooth`, `effectiveness_per_case`  

---

## Reproducibility & modeling notes

- **Diagnostics potentially counted twice (by design):**  
  With the current defaults, *diagnostic costs are added per case*, and **TP_0.0/FN include examination (and TP_0.0 adds radiography)**. This reproduces your original assumptions. If you want to **count diagnostics only once per case**, factor them out of TP/FN blocks.

- **Shares validation:**  
  Decision shares are validated to be in \[0,100] and to sum ≈100% (rows of all zeros are allowed).

- **Per-tooth vs per-case:**  
  Expected cost per tooth is mixed by `disease_prevalence`. Per-case cost is `teeth_count * cost_per_tooth + diagnostic_cost`.

- **AI usage:**  
  `ai_diagnosis` may be {0,1} **or** a probability in \[0,1]. If you want strict binary behavior, cast to int and clip.

---

## Project layout

```
ce-radiography/
├─ src/ce_radiography/
│  ├─ __init__.py
│  └─ model.py                 # Cost, ModelConfig, CostEffectivenessCalculator
├─ tests/
│  └─ test_model.py            # minimal smoke tests
├─ pyproject.toml              # Black/Ruff config
├─ README.md
└─ LICENSE
```

---

## Development (Black/Ruff/pytest)

Format & lint:

```bash
ruff check . --fix
black .
```

Run tests:

```bash
pytest -q
```

Optional pre-commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.6.8
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
```

---

## Cite

If you use this code in a paper, please cite your manuscript and this repository. Example BibTeX (edit with your details):

```bibtex

```

---

## License

Specify your license (e.g., MIT). Example:

```
MIT License — see LICENSE for details.
```
