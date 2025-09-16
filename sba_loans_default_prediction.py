#SBA Loan Default Prediction (Full Training Pipeline)

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Step 0: Set global seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Step 1: Load and preprocess dataset
# -------------------------------------
df = pd.read_csv("SBA_loans_project_2.csv")
df = df.dropna(subset=['MIS_Status'])
df['MIS_Status'] = df['MIS_Status'].astype(str).str.strip()
df = df[df['MIS_Status'].isin(['0', '1'])].copy()
df['MIS_Status'] = df['MIS_Status'].astype(int)
print(f"✅ Total loaded rows after filtering: {df.shape[0]}")

# Step 2: Define feature engineering function
# -------------------------------------------
def engineer_features(df):
    df = df.copy()
    for col in ['DisbursementGross', 'GrAppv', 'SBA_Appv']:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', np.nan).astype(float)

    df['DisbursementRatio'] = df['DisbursementGross'] / df['GrAppv'].replace(0, 1e-6)
    df['AppvRatio'] = df['SBA_Appv'] / df['GrAppv'].replace(0, 1e-6)
    df['LogDisbursement'] = np.log1p(df['DisbursementGross'])
    df['LogEmployees'] = np.log1p(df['NoEmp'])
    df['LogNewExist'] = df['NewExist'].replace({'0': 0, '1': 1, '2': 2}).astype(float)

    if 'ApprovalDate' in df.columns and 'DisbursementDate' in df.columns:
        df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'], errors='coerce')
        df['DisbursementDate'] = pd.to_datetime(df['DisbursementDate'], errors='coerce')
        df['ProcessingDays'] = (df['DisbursementDate'] - df['ApprovalDate']).dt.days
        df['ApprovalYear'] = df['ApprovalDate'].dt.year
        df['DisbursementYear'] = df['DisbursementDate'].dt.year
    else:
        df['ProcessingDays'] = np.nan
        df['ApprovalYear'] = np.nan
        df['DisbursementYear'] = np.nan

    df['EfficientProcessing'] = df['DisbursementRatio'] * df['ProcessingDays'].fillna(0)
    df['LoanToProcessingTime'] = df['GrAppv'] / (df['ProcessingDays'].fillna(1) + 1)
    df['HasZip'] = df['Zip'].notnull().astype(int)
    df['ZipLength'] = df['Zip'].astype(str).str.len()
    df['HasBank'] = df['Bank'].notnull().astype(int)
    df['BankStateMatch'] = (df['State'] == df['BankState']).astype(int)
    df['LoanToAssetRatio'] = df['GrAppv'] / df['NoEmp'].replace(0, 1e-6)
    df['DebtPerEmployee'] = df['GrAppv'] / df['NoEmp'].replace(0, 1e-6)
    df['LoanToZipRatio'] = df['GrAppv'] / (df['ZipLength'] + 1)
    df['IsUrban'] = (df['UrbanRural'] == 1).astype(int)
    df['IsNewBusiness'] = (df['NewExist'] == 2).astype(int)
    df['EmpLoanInteraction'] = df['GrAppv'] * df['NoEmp']
    df['LogAssetsPerEmployee'] = np.log1p(df['SBA_Appv'] / (df['NoEmp'] + 1))
    df['TimeSinceApproval'] = 2025 - df['ApprovalYear'].fillna(2025)

    return df

# Step 3: Define categorical encoding
# ------------------------------------
def encode_categoricals(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

# Step 4: Prepare features and labels
# -----------------------------------
X = engineer_features(df.drop(columns=['MIS_Status']))
X = encode_categoricals(X)
y = df['MIS_Status']
feature_columns = X.columns.tolist()

# Step 5: Split into train and test sets
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=RANDOM_STATE)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Step 6: Set hyperparameters and train model
# -------------------------------------------
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'lambda': 5,
    'alpha': 1,
    'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    'seed': RANDOM_STATE
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=50,
    verbose_eval=50
)

# Step 7: Find best F1 threshold
# ------------------------------
y_pred_proba = model.predict(dtest)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = f1_scores.max()
aucpr = average_precision_score(y_test, y_pred_proba)
print(f"Best F1 Threshold: {best_threshold}")
print(f"AUCPR: {aucpr}")
print(f"F1: {best_f1}")

# Step 8: Model interpretation (SHAP)
# -----------------------------------
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")


# Step 9: Confusion matrix at best threshold
# ------------------------------------------
y_pred_label = (y_pred_proba >= best_threshold).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_label))

# Step 10: Save final artifacts
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
model.save_model("artifacts/xgboost_model.json")
joblib.dump(best_threshold, "artifacts/f1_threshold.pkl")
joblib.dump(feature_columns, "artifacts/feature_columns.pkl")
print("✅ Artifacts saved.")
