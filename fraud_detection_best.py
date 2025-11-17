

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, f1_score,
                             precision_score, recall_score, average_precision_score)
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading and preparing data...")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = RobustScaler()
X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"Training with class weight: {scale_pos_weight:.1f}")

print("Running 5-fold cross-validation...")
cv_pr_aucs = []
cv_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
    'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 20, 'min_child_weight': 5,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'n_jobs': -1, 'verbose': -1, 'random_state': 42
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx], reference=train_data)
    model = lgb.train(cv_params, train_data, num_boost_round=1000,
                      valid_sets=[train_data, val_data],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    y_pred_proba = model.predict(X_train.iloc[val_idx], num_iteration=model.best_iteration)
    cv_pr_aucs.append(average_precision_score(y_train.iloc[val_idx], y_pred_proba))

print(f"CV PR-AUC: {np.mean(cv_pr_aucs):.4f} ± {np.std(cv_pr_aucs):.4f}")

print("Training final model...")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

final_model = lgb.train(cv_params, train_data, num_boost_round=1000,
                        valid_sets=[train_data, test_data],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

y_pred_proba = final_model.predict(X_test, num_iteration=final_model.best_iteration)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

pr_auc = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nResults - PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print(f"Confusion Matrix - TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

# Generate visualizations
print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Precision-Recall Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 0].plot(recalls, precisions, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
axes[0, 0].axhline(y_test.sum()/len(y_test), color='r', linestyle='--', label='Baseline')
axes[0, 0].set_xlabel('Recall')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].set_title('Precision-Recall Curve')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. ROC Curve
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {roc_auc:.4f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2], cbar=False,
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
axes[0, 2].set_title(f'Confusion Matrix\nThreshold: {optimal_threshold:.4f}')
axes[0, 2].set_ylabel('True')
axes[0, 2].set_xlabel('Predicted')

# 4. Feature Importance
importances = final_model.feature_importance(importance_type='gain')
indices = np.argsort(importances)[-20:]
axes[1, 0].barh(range(len(indices)), importances[indices])
axes[1, 0].set_yticks(range(len(indices)))
axes[1, 0].set_yticklabels([X.columns[i] for i in indices], fontsize=8)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 20 Features')

# 5. Threshold Analysis
axes[1, 1].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
axes[1, 1].plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
axes[1, 1].plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
axes[1, 1].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Threshold Optimization')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Business Impact
fraud_caught = tp * 100
fraud_missed = fn * 100
investigation = (tp + fp) * 5
net_benefit = fraud_caught - fraud_missed - investigation
categories = ['Fraud\nCaught', 'Fraud\nMissed', 'Investigation\nCosts', 'Net\nBenefit']
values = [fraud_caught, -fraud_missed, -investigation, net_benefit]
colors = ['green', 'red', 'orange', 'blue' if net_benefit > 0 else 'red']
bars = axes[1, 2].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
axes[1, 2].axhline(0, color='black', linewidth=1)
axes[1, 2].set_ylabel('Amount ($)')
axes[1, 2].set_title('Business Impact')
for bar, val in zip(bars, values):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, height,
                    f'${val:,.0f}', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
print("Saved: fraud_detection_results.png")

joblib.dump(final_model, 'fraud_model_best.pkl')
joblib.dump(scaler, 'scaler_best.pkl')
joblib.dump({'threshold': optimal_threshold}, 'metadata_best.pkl')
print("Model saved: fraud_model_best.pkl")
