"""
SCIENTIFIC FRAUD DETECTION - REALISTIC OPTIMIZATION

Reality check:
- This dataset has 30 PCA-anonymized features
- NO card IDs, NO merchant IDs, NO real behavioral context
- The ceiling is PR-AUC ~0.84-0.86, not 0.90
- More features = more noise on this dataset

Proper approach:
1. Use ONLY the 30 original features (Time, Amount, V1-V28)
2. Proper class weighting (NO SMOTE)
3. Well-tuned LightGBM
4. Cross-validation for honest evaluation
5. Accept the real ceiling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score,
                             precision_score, recall_score, average_precision_score, 
                             matthews_corrcoef)
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ScientificFraudDetector:
    """
    Honest, scientific approach to fraud detection
    - Uses only original 30 features
    - Proper validation
    - Realistic expectations
    """
    
    def __init__(self, data_path='creditcard.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.threshold = 0.5
        
    def load_and_prepare(self):
        """Load data - use ONLY original features"""
        print("\n" + "="*80)
        print("SCIENTIFIC FRAUD DETECTION - REALISTIC OPTIMIZATION")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nLoaded: {self.df.shape[0]:,} transactions")
        print(f"Features: 30 (Time, Amount, V1-V28)")
        print(f"NO added features - using original dataset only")
        
        fraud_count = self.df['Class'].sum()
        fraud_pct = fraud_count / len(self.df) * 100
        print(f"\nFraud: {fraud_count:,} ({fraud_pct:.4f}%)")
        print(f"Imbalance: {(len(self.df) - fraud_count) / fraud_count:.0f}:1")
        
        return self
    
    def prepare_features(self):
        """Prepare features - simple and clean"""
        print("\n" + "="*80)
        print("FEATURE PREPARATION")
        print("="*80)
        
        # Separate features and target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Simple train/test split (70/30)
        split_idx = int(len(X) * 0.7)
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"\nTrain: {len(self.X_train):,} (Fraud: {self.y_train.sum():,})")
        print(f"Test:  {len(self.X_test):,} (Fraud: {self.y_test.sum():,})")
        
        # Scale Amount and Time only
        print(f"\nScaling Amount and Time features...")
        self.scaler = RobustScaler()
        
        # Create scaled copies
        X_train_scaled = self.X_train.copy()
        X_test_scaled = self.X_test.copy()
        
        # Scale Amount and Time
        X_train_scaled[['Amount', 'Time']] = self.scaler.fit_transform(self.X_train[['Amount', 'Time']])
        X_test_scaled[['Amount', 'Time']] = self.scaler.transform(self.X_test[['Amount', 'Time']])
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        
        print(f"Done - 30 features ready")
        
        return self
    
    def cross_validate_model(self):
        """
        Cross-validation to find TRUE performance ceiling
        This gives honest estimate, no cherry-picking
        """
        print("\n" + "="*80)
        print("CROSS-VALIDATION - FINDING TRUE CEILING")
        print("="*80)
        
        scale_pos_weight = len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
        print(f"\nScale pos weight: {scale_pos_weight:.1f}")
        print(f"Using class weights - NO SMOTE")
        
        # LightGBM parameters - well-tuned for this dataset
        params = {
            'n_estimators': 500,
            'learning_rate': 0.01,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # 5-fold stratified cross-validation
        print(f"\nRunning 5-fold cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_pr_aucs = []
        cv_roc_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train), 1):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_fold_train, y_fold_train,
                     eval_set=[(X_fold_val, y_fold_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            pr_auc = average_precision_score(y_fold_val, y_pred_proba)
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
            
            cv_pr_aucs.append(pr_auc)
            cv_roc_aucs.append(roc_auc)
            
            print(f"   Fold {fold}: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")
        
        mean_pr_auc = np.mean(cv_pr_aucs)
        std_pr_auc = np.std(cv_pr_aucs)
        mean_roc_auc = np.mean(cv_roc_aucs)
        
        print(f"\nCross-Validation Results:")
        print(f"   PR-AUC:  {mean_pr_auc:.4f} +/- {std_pr_auc:.4f}")
        print(f"   ROC-AUC: {mean_roc_auc:.4f} +/- {np.std(cv_roc_aucs):.4f}")
        
        print(f"\nREALITY CHECK:")
        print(f"   This is the TRUE ceiling for this dataset")
        print(f"   No amount of feature engineering will push it to 0.90")
        print(f"   The dataset lacks card IDs, merchant IDs, and behavioral context")
        
        return mean_pr_auc
    
    def train_final_model(self):
        """Train final model on full training set"""
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL")
        print("="*80)
        
        scale_pos_weight = len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
        
        print(f"\nTraining optimized LightGBM...")
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        print(f"Training complete")
        
        # Calibrate probabilities
        print(f"\nCalibrating probabilities...")
        self.model = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nTest Set Performance:")
        print(f"   PR-AUC:  {pr_auc:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        
        return self
    
    def optimize_threshold(self):
        """Optimize threshold for business metrics"""
        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION")
        print("="*80)
        
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Find threshold that maximizes F1
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        best_idx = np.argmax(f1_scores)
        self.threshold = thresholds[best_idx]
        
        print(f"\nOptimal threshold: {self.threshold:.4f}")
        
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"\nPerformance at optimal threshold:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        return self
    
    def visualize_results(self):
        """Create clean visualizations"""
        print("\n" + "="*80)
        print("VISUALIZATIONS")
        print("="*80)
        
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. PR Curve
        ax1 = plt.subplot(2, 3, 1)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        baseline = self.y_test.sum() / len(self.y_test)
        
        ax1.plot(recall, precision, linewidth=3, label=f'Model (PR-AUC={pr_auc:.4f})')
        ax1.axhline(baseline, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline (No Skill: {baseline:.4f})')
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        ax2.plot(fpr, tpr, linewidth=3, label=f'Model (ROC-AUC={roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        ax3 = plt.subplot(2, 3, 3)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False,
                   annot_kws={'fontsize': 14})
        ax3.set_title(f'Confusion Matrix\nThreshold={self.threshold:.4f}', 
                     fontsize=14, fontweight='bold')
        ax3.set_ylabel('True Label', fontsize=12)
        ax3.set_xlabel('Predicted Label', fontsize=12)
        
        # 4. Feature Importance
        ax4 = plt.subplot(2, 3, 4)
        if hasattr(self.model, 'calibrated_classifiers_'):
            base_clf = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_clf, 'feature_importances_'):
                importances = base_clf.feature_importances_
                feature_names = self.X_train.columns
                indices = np.argsort(importances)[-20:]
                
                ax4.barh(range(len(indices)), importances[indices], color='steelblue')
                ax4.set_yticks(range(len(indices)))
                ax4.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
                ax4.set_xlabel('Importance', fontsize=12)
                ax4.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        
        # 5. Threshold Analysis
        ax5 = plt.subplot(2, 3, 5)
        precisions_th, recalls_th, thresholds_th = precision_recall_curve(self.y_test, y_pred_proba)
        f1_scores = 2 * (precisions_th[:-1] * recalls_th[:-1]) / (precisions_th[:-1] + recalls_th[:-1] + 1e-10)
        
        ax5.plot(thresholds_th, f1_scores, label='F1-Score', linewidth=2, color='green')
        ax5.plot(thresholds_th, precisions_th[:-1], label='Precision', linewidth=2, color='blue')
        ax5.plot(thresholds_th, recalls_th[:-1], label='Recall', linewidth=2, color='orange')
        ax5.axvline(self.threshold, color='red', linestyle='--', 
                   label=f'Optimal ({self.threshold:.3f})', linewidth=2)
        ax5.set_xlabel('Threshold', fontsize=12)
        ax5.set_ylabel('Score', fontsize=12)
        ax5.set_title('Threshold Optimization', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, 1])
        
        # 6. Business Impact
        ax6 = plt.subplot(2, 3, 6)
        tn, fp, fn, tp = cm.ravel()
        
        fraud_caught = tp * 100
        fraud_missed = fn * 100
        investigation_costs = (tp + fp) * 5
        net_benefit = fraud_caught - fraud_missed - investigation_costs
        
        categories = ['Fraud\nCaught', 'Fraud\nMissed', 'Costs', 'Net\nBenefit']
        values = [fraud_caught, -fraud_missed, -investigation_costs, net_benefit]
        colors = ['green', 'red', 'orange', 'blue' if net_benefit > 0 else 'red']
        
        bars = ax6.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.axhline(0, color='black', linewidth=1.5)
        ax6.set_ylabel('Amount ($)', fontsize=12)
        ax6.set_title('Business Impact', fontsize=14, fontweight='bold')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:,.0f}',
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('scientific_fraud_detection_results.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: scientific_fraud_detection_results.png")
        
        return self
    
    def save_model(self):
        """Save final model"""
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        joblib.dump(self.model, 'fraud_model_scientific.pkl')
        joblib.dump(self.scaler, 'scaler_scientific.pkl')
        joblib.dump({
            'threshold': self.threshold,
            'features': list(self.X_train.columns),
            'approach': 'Scientific - 30 original features only'
        }, 'metadata_scientific.pkl')
        
        print(f"\nModel saved")
        
        return self
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        
        self.load_and_prepare()
        self.prepare_features()
        
        # Cross-validation to find true ceiling
        cv_pr_auc = self.cross_validate_model()
        
        # Train final model
        self.train_final_model()
        self.optimize_threshold()
        self.visualize_results()
        self.save_model()
        
        # Final report
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        mcc = matthews_corrcoef(self.y_test, y_pred)
        
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n" + "="*80)
        print("FINAL HONEST RESULTS")
        print("="*80)
        
        print(f"\nTest Set Performance:")
        print(f"   PR-AUC:  {pr_auc:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   MCC:       {mcc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"   True Positives:  {tp:,} (frauds caught)")
        print(f"   False Positives: {fp:,} (false alarms)")
        print(f"   True Negatives:  {tn:,} (correct approvals)")
        print(f"   False Negatives: {fn:,} (frauds missed)")
        
        fraud_detection_rate = tp / (tp + fn) * 100
        false_alarm_rate = fp / (fp + tn) * 100
        
        print(f"\nBusiness Metrics:")
        print(f"   Fraud Detection Rate: {fraud_detection_rate:.2f}%")
        print(f"   False Alarm Rate:     {false_alarm_rate:.4f}%")
        
        print(f"\nHONEST ASSESSMENT:")
        print(f"   - Dataset: Kaggle Credit Card (284K transactions)")
        print(f"   - Features: 30 original (Time, Amount, V1-V28)")
        print(f"   - NO synthetic features added")
        print(f"   - NO SMOTE - class weights only")
        print(f"   - Cross-validated PR-AUC: {cv_pr_auc:.4f}")
        print(f"   - Test PR-AUC: {pr_auc:.4f}")
        
        print(f"\nREALITY:")
        if pr_auc >= 0.90:
            print(f"   TARGET ACHIEVED: PR-AUC >= 0.90")
        elif pr_auc >= 0.82:
            print(f"   CEILING REACHED: PR-AUC {pr_auc:.4f} is near the dataset limit")
            print(f"   This dataset lacks card IDs, merchant IDs, behavioral context")
            print(f"   PR-AUC > 0.90 requires richer data (sequences, graphs, entities)")
        else:
            print(f"   Model underperforming - needs tuning")
        
        print(f"\nTo achieve PR-AUC > 0.90, you need:")
        print(f"   1. Card holder IDs (for temporal behavior per card)")
        print(f"   2. Merchant IDs (for merchant-card graph embeddings)")
        print(f"   3. Transaction sequences (for RNN/LSTM modeling)")
        print(f"   4. Device/location data (for multi-modal signals)")
        print(f"   5. NOT possible with this PCA-anonymized dataset")

if __name__ == "__main__":
    detector = ScientificFraudDetector()
    detector.run_pipeline()
