# Cost-Effectiveness of AI-Assisted Detection of Apical Periodontitis (Paper-code)

## Abstract
Artificial intelligence (AI) is transforming medical imaging, yet its economic impact in dentistry remains largely unexplored. This study evaluated the cost-effectiveness of AI-assisted detection of apical periodontitis on panoramic radiographs, including downstream clinical decision-making. Using data from a randomized study on AI-assisted detection of apical lesions, a decision-analytic model was established to analyse costs and effectiveness from a German mixed-payer perspective. AI support reduced average costs per case and increased treatment effectiveness, outperforming unaided examiner performance. These gains were primarily driven by improved specificity, reducing false-positive detection. However, effects varied by examiner experience; junior clinicians achieved the greatest cost savings and effectiveness gains, whereas senior examiners showed reduced sensitivity and slightly lower effectiveness at similar costs. AI-assisted diagnostics offer significant potential to improve cost-effectiveness by reducing overtreatment, with benefits being most pronounced among less experienced practitioners. Adapting AI systems to individual examiners or experience levels might further enhance clinical and economic impact. 

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
git clone EWalter-lab/cea-ai_radiography && cd cea-ai_radiography

# install package (editable) + dev tools
pip install -e .
```

---

## Quickstart

```python
from cea_radiography import performance, cea
import pandas as pd

filepath = "../01_Data/1_Data/master_dataset.csv"
df= pd.read_csv(filepath)

eval=performance.PerformanceConfig(
        ground_truth_col="ground_truth",
        diagnosis_col="diagnosis",
        groupby_cols=("examiner","ai_diagnosis"),
        passthrough_cols=("experience_level",),
        passthrough_reduce="mode",
        positive_label=1,
        negative_label=0,
        zero_division_value=0.0,
        enforce_binary=True
    )

evaluator = performance.PerformanceEvaluator(eval)
performance_df=evaluator.evaluate(df)
```

```python
from cea_radiography import performance, cea
import pandas as pd

Cost_EUR = cea.Cost(
    intraoral_radiography = 14.64,
    cbct= 214.00,
    ai = 8,
    rct = 415.67,
    panoramic = 43.92,
    crown = 377.27,
    examination = 21.96)

calculator = cea.CostEffectivenessCalculator(Cost_EUR)
calculator.fit_cost(performance_df)
calculator.fit_effectiveness()

df=calculator.result_
```

---

## Pathway semantics

### Cost (current defaults)

- **TP_0.0**: examination + radiography + RCT + crown  
- **TP_1.0**: intraoral_radiography + RCT + crown  
- **TP_2.0**: RCT + crown  

- **FP_0.0**: 0  
- **FP_1.0**: intraoral_radiography
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
```

---

## Project layout

```
ce-radiography/
├─ src/ce_radiography/
│  ├─ __init__.py
│  ├─ performance.py    # PerformanceConfig, PerformanceEvaluator
│  ├─ evaluator.py    
│  └─ cea.py            # Cost, ModelConfig, CostEffectivenessCalculator
├─ pyproject.toml
├─ README.md
└─ LICENSE
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
