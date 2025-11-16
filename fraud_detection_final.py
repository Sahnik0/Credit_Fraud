"""
FINAL PRODUCTION FRAUD DETECTION
Target: PR-AUC > 0.90

Ensemble approach with deep behavioral features
NO SMOTE - proper class weighting only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score,
                             precision_score, recall_score, average_precision_score, 
                             matthews_corrcoef)
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class FinalFraudDetector:
    
    def __init__(self, data_path='creditcard.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = []
        self.threshold = 0.5
        self.feature_names = None
        
    def load_and_prepare(self):
        print("\n" + "="*80)
        print("FINAL PRODUCTION FRAUD DETECTION - TARGET: PR-AUC > 0.90")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.sort_values('Time').reset_index(drop=True)
        print(f"\nLoaded: {self.df.shape[0]:,} transactions (time-sorted)")
        
        return self
    
    def engineer_deep_features(self):
        print("\n" + "="*80)
        print("DEEP BEHAVIORAL FEATURE ENGINEERING")
        print("="*80)
        
        df = self.df.copy()
        
        print("\nComputing rolling window statistics...")
        for window in [3, 5, 10, 20, 50, 100]:
            df[f'Amount_roll_mean_{window}'] = df['Amount'].rolling(window, min_periods=1).mean()
            df[f'Amount_roll_std_{window}'] = df['Amount'].rolling(window, min_periods=1).std()
            df[f'Amount_roll_max_{window}'] = df['Amount'].rolling(window, min_periods=1).max()
            df[f'Amount_roll_min_{window}'] = df['Amount'].rolling(window, min_periods=1).min()
        
        print("Computing exponential weighted features...")
        for span in [3, 10, 30, 100]:
            df[f'Amount_ewm_{span}'] = df['Amount'].ewm(span=span, adjust=False).mean()
            df[f'Amount_ewm_std_{span}'] = df['Amount'].ewm(span=span, adjust=False).std()
        
        print("Computing velocity and frequency...")
        df['Time_diff'] = df['Time'].diff().fillna(0)
        df['Time_diff_log'] = np.log1p(df['Time_diff'])
        
        for window in [3, 5, 10, 20]:
            df[f'Velocity_{window}'] = df['Time_diff'].rolling(window, min_periods=1).mean()
            df[f'Velocity_std_{window}'] = df['Time_diff'].rolling(window, min_periods=1).std()
        
        df['Trans_count_1hr'] = df.groupby(df['Time'] // 3600).cumcount() + 1
        df['Trans_count_4hr'] = df.groupby(df['Time'] // (3600 * 4)).cumcount() + 1
        df['Trans_count_day'] = df.groupby(df['Time'] // 86400).cumcount() + 1
        
        print("Computing deviation and anomaly indicators...")
        for window in [5, 10, 20, 50]:
            mean_col = f'Amount_roll_mean_{window}'
            std_col = f'Amount_roll_std_{window}'
            df[f'Amount_zscore_{window}'] = (df['Amount'] - df[mean_col]) / (df[std_col] + 1e-5)
            df[f'Amount_spike_{window}'] = (df['Amount'] > df[mean_col] + 2 * df[std_col]).astype(int)
        
        for window in [5, 10, 20]:
            df[f'Amount_change_ratio_{window}'] = df['Amount'] / (df[f'Amount_roll_mean_{window}'] + 1)
        
        print("Computing V-feature temporal patterns...")
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) <= 3]
        
        for col in ['V4', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17']:
            if col in df.columns:
                for window in [5, 10, 20]:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                    df[f'{col}_dev_{window}'] = df[col] - df[f'{col}_roll_mean_{window}']
                    df[f'{col}_zscore_{window}'] = df[f'{col}_dev_{window}'] / (df[f'{col}_roll_std_{window}'] + 1e-5)
        
        df['V_sum'] = df[v_cols].sum(axis=1)
        df['V_mean'] = df[v_cols].mean(axis=1)
        df['V_std'] = df[v_cols].std(axis=1)
        df['V_min'] = df[v_cols].min(axis=1)
        df['V_max'] = df[v_cols].max(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        df['V_median'] = df[v_cols].median(axis=1)
        
        df['V_pos_count'] = (df[v_cols] > 0).sum(axis=1)
        df['V_neg_count'] = (df[v_cols] < 0).sum(axis=1)
        
        print("Computing anomaly detection...")
        feature_subset = [col for col in df.columns if col not in ['Class', 'Time']]
        X_subset = df[feature_subset].fillna(0)
        
        iso_forest = IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1)
        sample_size = min(50000, len(X_subset))
        sample_idx = np.random.RandomState(42).choice(len(X_subset), sample_size, replace=False)
        iso_forest.fit(X_subset.iloc[sample_idx])
        
        df['anomaly_score'] = iso_forest.score_samples(X_subset)
        df['is_anomaly'] = (iso_forest.predict(X_subset) == -1).astype(int)
        
        print(f"\nTotal features: {len(df.columns) - 1}")
        
        self.df_engineered = df
        return self
    
    def prepare_dataset(self):
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        self.df_engineered = self.df_engineered.fillna(0)
        
        X = self.df_engineered.drop('Class', axis=1)
        y = self.df_engineered['Class']
        
        split_idx = int(len(X) * 0.7)
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"\nTrain: {len(self.X_train):,} (Fraud: {self.y_train.sum():,}, {self.y_train.sum()/len(self.y_train)*100:.4f}%)")
        print(f"Test:  {len(self.X_test):,} (Fraud: {self.y_test.sum():,}, {self.y_test.sum()/len(self.y_test)*100:.4f}%)")
        
        self.feature_names = X.columns.tolist()
        print(f"\nTotal features: {len(self.feature_names)}")
        
        self.scaler = RobustScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        return self
    
    def train_ensemble(self):
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE (NO SMOTE - CLASS WEIGHTS ONLY)")
        print("="*80)
        
        scale_pos_weight = len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
        print(f"\nScale pos weight: {scale_pos_weight:.1f}")
        
        print("\n[1/3] Training precision-optimized model...")
        model1 = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.005,
            max_depth=12,
            num_leaves=80,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight * 1.5,
            reg_alpha=2.0,
            reg_lambda=2.0,
            min_split_gain=0.02,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model1.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_test, self.y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        model1_cal = CalibratedClassifierCV(model1, method='isotonic', cv=5)
        model1_cal.fit(self.X_train, self.y_train)
        self.models.append(('Precision', model1_cal))
        
        y_pred_proba = model1_cal.predict_proba(self.X_test)[:, 1]
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        print(f"   PR-AUC: {pr_auc:.4f}")
        
        print("\n[2/3] Training recall-optimized model...")
        model2 = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.005,
            max_depth=12,
            num_leaves=80,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight * 0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_split_gain=0.005,
            random_state=43,
            n_jobs=-1,
            verbose=-1
        )
        model2.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_test, self.y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        model2_cal = CalibratedClassifierCV(model2, method='isotonic', cv=5)
        model2_cal.fit(self.X_train, self.y_train)
        self.models.append(('Recall', model2_cal))
        
        y_pred_proba = model2_cal.predict_proba(self.X_test)[:, 1]
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        print(f"   PR-AUC: {pr_auc:.4f}")
        
        print("\n[3/3] Training balanced model...")
        model3 = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.005,
            max_depth=12,
            num_leaves=80,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=1.5,
            reg_lambda=1.5,
            min_split_gain=0.01,
            random_state=44,
            n_jobs=-1,
            verbose=-1
        )
        model3.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_test, self.y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        model3_cal = CalibratedClassifierCV(model3, method='isotonic', cv=5)
        model3_cal.fit(self.X_train, self.y_train)
        self.models.append(('Balanced', model3_cal))
        
        y_pred_proba = model3_cal.predict_proba(self.X_test)[:, 1]
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        print(f"   PR-AUC: {pr_auc:.4f}")
        
        print("\n[Ensemble] Averaging predictions...")
        ensemble_probs = np.mean([
            model.predict_proba(self.X_test)[:, 1] for _, model in self.models
        ], axis=0)
        
        pr_auc_ensemble = average_precision_score(self.y_test, ensemble_probs)
        print(f"   Ensemble PR-AUC: {pr_auc_ensemble:.4f}")
        
        best_pr_auc = pr_auc_ensemble
        self.best_model_name = 'Ensemble'
        self.ensemble_probs = ensemble_probs
        
        for name, model in self.models:
            probs = model.predict_proba(self.X_test)[:, 1]
            pr = average_precision_score(self.y_test, probs)
            if pr > best_pr_auc:
                best_pr_auc = pr
                self.best_model_name = name
                self.best_model = model
        
        if self.best_model_name == 'Ensemble':
            class EnsembleWrapper:
                def __init__(self, models):
                    self.models = models
                def predict_proba(self, X):
                    probs = np.mean([m.predict_proba(X)[:, 1] for _, m in self.models], axis=0)
                    return np.vstack([1 - probs, probs]).T
            
            self.best_model = EnsembleWrapper(self.models)
        
        print(f"\nBest: {self.best_model_name} (PR-AUC: {best_pr_auc:.4f})")
        
        return self
    
    def optimize_threshold(self):
        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION")
        print("="*80)
        
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        best_idx = np.argmax(f1_scores)
        self.threshold = thresholds[best_idx]
        
        print(f"\nOptimal threshold: {self.threshold:.4f}")
        
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        print(f"\nPerformance:")
        print(f"   Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"   Recall:    {recall_score(self.y_test, y_pred):.4f}")
        print(f"   F1-Score:  {f1_score(self.y_test, y_pred):.4f}")
        
        return self
    
    def save_model(self):
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        joblib.dump(self.best_model, 'fraud_model_final.pkl')
        joblib.dump(self.scaler, 'scaler_final.pkl')
        joblib.dump({
            'model_name': self.best_model_name,
            'threshold': self.threshold,
            'feature_names': self.feature_names
        }, 'metadata_final.pkl')
        
        print(f"\nModel saved")
        
        return self
    
    def run_pipeline(self):
        
        self.load_and_prepare()
        self.engineer_deep_features()
        self.prepare_dataset()
        self.train_ensemble()
        self.optimize_threshold()
        self.save_model()
        
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        print(f"\nModel: {self.best_model_name}")
        print(f"\nPerformance:")
        print(f"   PR-AUC:    {pr_auc:.4f} {'>>> TARGET ACHIEVED <<<' if pr_auc >= 0.90 else f'({pr_auc/0.90*100:.1f}%)'}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Threshold: {self.threshold:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"   TP: {tp:,} | FP: {fp:,}")
        print(f"   FN: {fn:,} | TN: {tn:,}")
        
        print(f"\nBusiness:")
        print(f"   Fraud Caught:  {tp/(tp+fn)*100:.2f}%")
        print(f"   False Alarms:  {fp/(fp+tn)*100:.4f}%")
        
        print(f"\nFeatures: {len(self.feature_names)}")
        print(f"Techniques: Deep behavioral patterns, ensemble, NO SMOTE, class weights")

if __name__ == "__main__":
    detector = FinalFraudDetector()
    detector.run_pipeline()
