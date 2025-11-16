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

## Alternative Scripts

### Scientific Approach (with visualizations)
```powershell
python fraud_detection_scientific.py
```
Outputs: `scientific_fraud_detection_results.png`

### Production Approach (behavioral features)
```powershell
python fraud_detection_production.py
```
Outputs: `production_fraud_detection_results.png`

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
├── fraud_detection_best.py                 # Best model (recommended)
├── fraud_detection_scientific.py           # With visualizations
├── fraud_detection_production.py           # Behavioral features attempt
├── fraud_model_best.pkl                    # Trained model
├── scaler_best.pkl                         # Feature scaler
├── metadata_best.pkl                       # Model metadata
├── HONEST_ASSESSMENT.md                    # Detailed analysis
└── .venv/                                  # Virtual environment
```

## Key Takeaways

1. **Use `fraud_detection_best.py`** for best performance
2. **Expected PR-AUC: 0.45-0.62** (this is the dataset ceiling)
3. **NO SMOTE** - uses class weights for imbalance
4. **30 features only** - Time, Amount, V1-V28 (original PCA features)
5. **Cross-validated** - honest performance estimation

## To Achieve PR-AUC > 0.90

You need fundamentally different data:
- Card holder IDs (for per-card behavior)
- Merchant IDs (for graph embeddings)
- Transaction sequences (for temporal modeling)
- Device/location metadata

