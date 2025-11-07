# SBA Loan Default Prediction (Full Training Pipeline)

A comprehensive machine learning pipeline designed to **predict the likelihood of Small Business Administration (SBA) loan default** using real-world loan data.  
The project focuses on full-data training, advanced feature engineering, interpretability (via SHAP), and reproducible model deployment artifacts.

---

## Overview
- **Goal:** Predict SBA loan default risk (`MIS_Status`) using borrower, loan, and disbursement attributes.
- **Approach:** End-to-end binary classification pipeline using **XGBoost**, optimized for **AUCPR** and **F1-score** thresholds.
- **Deliverables:**
  - Trained XGBoost model (`xgboost_model.json`)
  - Optimized classification threshold (`f1_threshold.pkl`)
  - Feature list (`feature_columns.pkl`)
  - SHAP explainability summary plot

---

## Methodology
### 🔹 1. Data Preprocessing
- Loaded raw dataset: `SBA_loans_project_2.csv`
- Filtered valid loan outcomes (`MIS_Status` = 0 or 1)
- Cleaned and standardized numeric fields (`DisbursementGross`, `GrAppv`, `SBA_Appv`)
- Handled missing data and converted categorical variables to encoded integers

### 🔹 2. Feature Engineering
Created over **15+ derived variables**, including:
- `DisbursementRatio`, `AppvRatio`, `LoanToProcessingTime`
- Log-transformed continuous variables (`LogDisbursement`, `LogEmployees`)
- Time-based features (`ApprovalYear`, `ProcessingDays`, `TimeSinceApproval`)
- Interaction terms (`EmpLoanInteraction`, `LoanToAssetRatio`)
- Flags and binary indicators (`IsUrban`, `HasBank`, `BankStateMatch`, etc.)

### 🔹 3. Model Training
- Model: **XGBoost (binary:logistic)**  
- Evaluation metric: **Area Under Precision-Recall Curve (AUCPR)**
- Parameters tuned for class imbalance:
  - `scale_pos_weight` = ratio of negatives to positives
  - `eta=0.03`, `max_depth=6`, `subsample=0.85`, `colsample_bytree=0.85`
- Early stopping (`early_stopping_rounds=50`) and reproducible seed (`random_state=42`)

### 🔹 4. Model Evaluation
- **F1-score optimization**: calculated best probability threshold using `precision_recall_curve`
- **Metrics:**
  - Best Threshold → printed dynamically after training
  - **AUCPR**, **Best F1**, and confusion matrix displayed
- Saved top metrics for reproducibility

### 🔹 5. Model Explainability
Used **SHAP (SHapley Additive exPlanations)** for post-hoc interpretability:
- Generated global **feature importance bar chart**
- Identified top factors driving defaults or successful repayments

---

## Output Artifacts
All saved under `/artifacts`:
| File | Description |
|------|--------------|
| `xgboost_model.json` | Trained XGBoost model |
| `f1_threshold.pkl` | Optimized decision threshold |
| `feature_columns.pkl` | List of final feature names used during training |

---

## How to Run

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/sba-loan-default-prediction.git
cd sba-loan-default-prediction

# 2. Create and activate environment
python -m venv .venv
source .venv/bin/activate     # On macOS/Linux
# .venv\Scripts\activate      # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset in project root
#    (expected file name: SBA_loans_project_2.csv)

# 5. Run training script / notebook
python Final_code.py
# or
jupyter notebook Final_code.ipynb
