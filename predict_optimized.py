"""
Optimized Fraud Prediction Script
Uses trained model with calibrated probabilities
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (classification_report, confusion_matrix, 
                             average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

class OptimizedFraudPredictor:
    """Production-ready fraud predictor with calibrated probabilities"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        
    def load_model(self):
        """Load trained model and preprocessing objects"""
        print("\n" + "="*80)
        print("LOADING OPTIMIZED FRAUD DETECTION MODEL")
        print("="*80)
        
        try:
            self.model = joblib.load('fraud_model_optimized.pkl')
            self.scaler = joblib.load('scaler_optimized.pkl')
            self.metadata = joblib.load('metadata_optimized.pkl')
            
            print(f"\nModel: {self.metadata['model_name']}")
            print(f"Threshold: {self.metadata['threshold']:.4f}")
            print(f"Features: {len(self.metadata['feature_names'])}")
            print(f"\nModel loaded successfully")
            
            return True
        except FileNotFoundError as e:
            print(f"\nError: Model files not found")
            print(f"Please run fraud_detection_optimized.py first to train the model")
            return False
    
    def _engineer_features(self, df):
        """Apply same feature engineering as training"""
        
        # Get V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        
        # 1. Statistical aggregations of V features
        df['V_sum'] = df[v_cols].sum(axis=1)
        df['V_mean'] = df[v_cols].mean(axis=1)
        df['V_std'] = df[v_cols].std(axis=1)
        df['V_min'] = df[v_cols].min(axis=1)
        df['V_max'] = df[v_cols].max(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        
        # 2. Interaction features
        if 'V14' in df.columns and 'V17' in df.columns:
            df['V14_V17'] = df['V14'] * df['V17']
        if 'V14' in df.columns and 'V12' in df.columns:
            df['V14_V12'] = df['V14'] * df['V12']
        if 'V10' in df.columns and 'V17' in df.columns:
            df['V10_V17'] = df['V10'] * df['V17']
        if 'V14' in df.columns and 'V10' in df.columns:
            df['V14_V10'] = df['V14'] * df['V10']
        
        # 3. Amount-based features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_squared'] = df['Amount'] ** 2
        
        # 4. Time-based features
        df['Time_hour'] = (df['Time'] / 3600) % 24
        df['Time_day'] = (df['Time'] / 86400).astype(int)
        
        # 5. Amount-Time interaction
        df['Amount_Time'] = df['Amount'] * df['Time']
        
        # 6. V feature counts and ratios
        df['V_positive_count'] = (df[v_cols] > 0).sum(axis=1)
        df['V_negative_count'] = (df[v_cols] < 0).sum(axis=1)
        df['V_zero_count'] = (df[v_cols] == 0).sum(axis=1)
        
        # 7. Extreme value indicators
        for col in ['V1', 'V2', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']:
            if col in df.columns:
                df[f'{col}_abs'] = df[col].abs()
                df[f'{col}_squared'] = df[col] ** 2
        
        return df
    
    def _preprocess(self, df):
        """Preprocess data for prediction"""
        
        # Remove Class column if present
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        
        # Engineer features
        df_engineered = self._engineer_features(df.copy())
        
        # Ensure feature order matches training
        feature_names = self.metadata['feature_names']
        
        # Add missing features with zeros
        for feature in feature_names:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0
        
        # Reorder to match training
        df_engineered = df_engineered[feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df_engineered)
        
        return pd.DataFrame(X_scaled, columns=feature_names, index=df.index)
    
    def predict(self, data_path):
        """Make predictions on new data"""
        print("\n" + "="*80)
        print("MAKING PREDICTIONS")
        print("="*80)
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"\nLoaded: {len(df):,} transactions")
        
        # Check if true labels exist
        has_labels = 'Class' in df.columns
        if has_labels:
            y_true = df['Class']
            print(f"True labels available: {(y_true==1).sum()} frauds")
        
        # Preprocess
        print(f"\nPreprocessing data...")
        X = self._preprocess(df)
        
        # Predict
        print(f"Generating predictions...")
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.metadata['threshold']).astype(int)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Transaction_ID': range(len(df)),
            'Fraud_Probability': y_pred_proba,
            'Prediction': y_pred,
            'Fraud_Label': ['FRAUD' if p == 1 else 'NORMAL' for p in y_pred],
            'Risk_Level': self._get_risk_level(y_pred_proba)
        })
        
        if has_labels:
            results['True_Label'] = y_true
        
        # Save results
        output_path = 'fraud_predictions_optimized.csv'
        results.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        
        # Summary
        print(f"\nPrediction Summary:")
        print(f"   Total transactions:     {len(results):,}")
        print(f"   Predicted FRAUD:        {(results['Prediction']==1).sum():,} ({(results['Prediction']==1).sum()/len(results)*100:.2f}%)")
        print(f"   Predicted NORMAL:       {(results['Prediction']==0).sum():,} ({(results['Prediction']==0).sum()/len(results)*100:.2f}%)")
        
        print(f"\nRisk Level Distribution:")
        for risk in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = (results['Risk_Level'] == risk).sum()
            pct = count / len(results) * 100
            print(f"   {risk:<10}: {count:>6,} ({pct:>5.2f}%)")
        
        # Evaluate if labels available
        if has_labels:
            print("\n" + "="*80)
            print("MODEL EVALUATION")
            print("="*80)
            
            self._evaluate_predictions(y_true, y_pred, y_pred_proba)
        
        return results
    
    def _get_risk_level(self, probabilities):
        """Categorize transactions by risk level"""
        risk_levels = []
        for prob in probabilities:
            if prob >= 0.90:
                risk_levels.append('CRITICAL')
            elif prob >= 0.70:
                risk_levels.append('HIGH')
            elif prob >= 0.40:
                risk_levels.append('MEDIUM')
            else:
                risk_levels.append('LOW')
        return risk_levels
    
    def _evaluate_predictions(self, y_true, y_pred, y_pred_proba):
        """Evaluate predictions against true labels"""
        
        # Calculate metrics
        pr_auc = average_precision_score(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nPerformance Metrics:")
        print(f"   PR-AUC:    {pr_auc:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix:")
        print(f"   True Negatives:  {tn:,}")
        print(f"   False Positives: {fp:,}")
        print(f"   False Negatives: {fn:,}")
        print(f"   True Positives:  {tp:,}")
        
        # Business metrics
        fraud_detected_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
        print(f"\nBusiness Metrics:")
        print(f"   Fraud Detection Rate: {fraud_detected_rate:.2f}%")
        print(f"   False Alarm Rate:     {false_alarm_rate:.2f}%")
        
        # Estimated business impact
        fraud_amount_avg = 100
        investigation_cost = 5
        
        fraud_caught = tp * fraud_amount_avg
        fraud_missed = fn * fraud_amount_avg
        investigation_costs = (tp + fp) * investigation_cost
        net_benefit = fraud_caught - investigation_costs - fraud_missed
        
        print(f"\nEstimated Business Impact:")
        print(f"   Fraud Caught:        ${fraud_caught:,.0f}")
        print(f"   Fraud Missed:        ${fraud_missed:,.0f}")
        print(f"   Investigation Costs: ${investigation_costs:,.0f}")
        print(f"   Net Benefit:         ${net_benefit:,.0f}")
    
    def predict_single_transaction(self, transaction_data):
        """Predict fraud for a single transaction"""
        
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame([transaction_data])
        
        # Preprocess
        X = self._preprocess(df)
        
        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        pred = int(prob >= self.metadata['threshold'])
        risk = self._get_risk_level([prob])[0]
        
        result = {
            'Fraud_Probability': prob,
            'Prediction': 'FRAUD' if pred == 1 else 'NORMAL',
            'Risk_Level': risk,
            'Recommendation': self._get_recommendation(prob, risk)
        }
        
        return result
    
    def _get_recommendation(self, probability, risk_level):
        """Generate action recommendation"""
        if risk_level == 'CRITICAL':
            return f"IMMEDIATE ACTION REQUIRED - Block transaction ({probability:.1%} fraud probability)"
        elif risk_level == 'HIGH':
            return f"Manual review recommended ({probability:.1%} fraud probability)"
        elif risk_level == 'MEDIUM':
            return f"Monitor transaction ({probability:.1%} fraud probability)"
        else:
            return f"Approve transaction ({probability:.1%} fraud probability)"

if __name__ == "__main__":
    # Create predictor
    predictor = OptimizedFraudPredictor()
    
    # Load model
    if predictor.load_model():
        # Make predictions on full dataset
        results = predictor.predict('creditcard.csv')
        
        print("\n" + "="*80)
        print("PREDICTION COMPLETE")
        print("="*80)
        print(f"\nResults saved to: fraud_predictions_optimized.csv")
