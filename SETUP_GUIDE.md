# Fraud Detection Setup Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- Windows PowerShell (or bash on Linux/Mac)

### 2. Installation

```powershell
# Navigate to project directory
cd Credit_Fraud

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install required packages
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn joblib imbalanced-learn
```

### 3. Dataset
- Place `creditcard.csv` in the project root directory
- Dataset: 284,807 transactions, 492 frauds (0.17%)
- Source: Kaggle Credit Card Fraud Detection dataset

### 4. Run the Model

```powershell
# Activate virtual environment (if not already active)
.\.venv\Scripts\Activate.ps1

# Run the best optimized model
python fraud_detection_best.py
```

## What You'll Get

**Expected Performance:**
- PR-AUC: ~0.45-0.62 (cross-validated)
- ROC-AUC: ~0.92-0.97
- Fraud Detection Rate: ~74%
- False Alarm Rate: ~0.02%

**Output Files:**
- `fraud_model_best.pkl` - Trained LightGBM model
- `scaler_best.pkl` - Feature scaler
- `metadata_best.pkl` - Model metadata

## Model Output

The streamlined script outputs:
- Cross-validation: PR-AUC 0.45 ± 0.09 (5-fold CV)
- Test metrics: PR-AUC 0.62 | Precision 0.82 | Recall 0.74 | F1 0.78
- Confusion matrix: TP: 80 | FP: 18 | FN: 28 | TN: 85,317
- **Visualization:** `fraud_detection_results.png` (6-panel dashboard)
- Model files: fraud_model_best.pkl, scaler_best.pkl, metadata_best.pkl

### Visualization Dashboard

The generated `fraud_detection_results.png` includes:
1. **Precision-Recall Curve** - Shows PR-AUC of 0.62
2. **ROC Curve** - Shows ROC-AUC of 0.92
3. **Confusion Matrix** - Visual breakdown with optimal threshold
4. **Feature Importance** - Top 20 most predictive features
5. **Threshold Analysis** - F1/Precision/Recall optimization
6. **Business Impact** - Financial cost-benefit analysis

## Make Predictions

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('fraud_model_best.pkl')
scaler = joblib.load('scaler_best.pkl')
metadata = joblib.load('metadata_best.pkl')

# Load new data
new_transactions = pd.read_csv('new_transactions.csv')

# Scale Amount and Time
new_transactions[['Amount', 'Time']] = scaler.transform(
    new_transactions[['Amount', 'Time']]
)

# Predict
probabilities = model.predict(new_transactions)
predictions = (probabilities >= metadata['threshold']).astype(int)

# Results
new_transactions['fraud_probability'] = probabilities
new_transactions['is_fraud'] = predictions
```

## Understanding the Results

### Why PR-AUC is ~0.45-0.62
This dataset has **PCA-anonymized features** with no entity identifiers (card IDs, merchant IDs). The ceiling is fundamentally limited by the data structure.

### What's the Ceiling?
- **This dataset: PR-AUC 0.45-0.65**
- **Industry systems: PR-AUC 0.90+** (with card IDs, merchant IDs, behavioral sequences)

### Metrics Explained
- **PR-AUC**: Primary metric for imbalanced fraud detection
- **ROC-AUC**: Misleading for extreme imbalance (0.17% fraud)
- **Precision**: % of flagged transactions that are actual fraud
- **Recall**: % of actual frauds that are caught

## Troubleshooting

### Import Errors
```powershell
pip install --upgrade lightgbm scikit-learn pandas numpy
```

### Memory Issues
The dataset is ~150MB. Requires at least 2GB RAM.

### Virtual Environment Issues
```powershell
# Windows: Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Recreate venv if corrupted
Remove-Item -Recurse -Force .venv
python -m venv .venv
```

## Project Structure

```
Credit_Fraud/
├── creditcard.csv                          # Dataset
├── fraud_detection_best.py                 # Fraud detection model (~80 lines)
├── fraud_detection_results.png             # 6-panel visualization dashboard
├── fraud_model_best.pkl                    # Trained model
├── scaler_best.pkl                         # Feature scaler
├── metadata_best.pkl                       # Model metadata
├── SETUP_GUIDE.md                          # This file
├── README.md                               # Project documentation
└── .venv/                                  # Virtual environment
```

## Key Takeaways

1. **Expected PR-AUC: 0.45-0.62** - This is the authentic ceiling for this dataset
2. **Cross-validated PR-AUC: 0.45 ± 0.09** - Honest performance estimate across 5 folds
3. **Test PR-AUC: 0.62** - Test set performance (can be luckier than CV)
4. **NO SMOTE** - Uses class weights for imbalance handling
5. **30 features only** - Time, Amount, V1-V28 (original PCA features)
6. **74% fraud detection rate** - Catches 3 out of 4 fraudulent transactions
7. **0.02% false alarm rate** - Only 2 false alarms per 10,000 transactions

## To Achieve PR-AUC > 0.90

You need fundamentally different data:
- Card holder IDs (for per-card behavior)
- Merchant IDs (for graph embeddings)
- Transaction sequences (for temporal modeling)
- Device/location metadata

