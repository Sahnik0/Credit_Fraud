"""
Proper Fraud Prediction Script - Production Ready
Uses the ROOT FIX model trained on Kaggle Credit Card dataset
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class ProperFraudPredictor:
    """Production-ready fraud predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.threshold = 0.5
    
    def load_model(self):
        """Load trained model and artifacts"""
        print("="*70)
        print("Loading Proper Fraud Detection Model...")
        print("="*70)
        
        try:
            self.model = joblib.load('fraud_model_proper.pkl')
            print("✓ Model loaded")
            
            self.scaler = joblib.load('scaler_proper.pkl')
            print("✓ Scaler loaded")
            
            self.metadata = joblib.load('metadata_proper.pkl')
            print("✓ Metadata loaded")
            
            self.threshold = self.metadata.get('threshold', 0.5)
            
            print(f"\n📊 Model Info:")
            print(f"   Type: {self.metadata['model_name']}")
            print(f"   Dataset: {self.metadata['dataset']}")
            print(f"   Optimal Threshold: {self.threshold:.2f}")
            print(f"   Features: {len(self.metadata['features'])}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nPlease run: python fraud_detection_proper.py")
            return False
    
    def predict(self, data_path=None, df=None):
        """
        Predict fraud on transaction data
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file with transaction data
        df : DataFrame
            DataFrame with transaction data
            
        Returns:
        --------
        DataFrame with predictions
        """
        if data_path is None and df is None:
            raise ValueError("Provide either data_path or df")
        
        # Load data
        if data_path:
            print(f"\n📂 Loading data: {data_path}")
            df = pd.read_csv(data_path)
        
        print(f"✓ Loaded {len(df):,} transactions")
        
        # Check if Class column exists (for evaluation)
        has_labels = 'Class' in df.columns
        if has_labels:
            y_true = df['Class'].copy()
            df = df.drop('Class', axis=1)
        else:
            y_true = None
        
        # Preprocess
        X = self._preprocess(df)
        
        # Predict
        print("\n🔍 Making predictions...")
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Create results
        results = pd.DataFrame({
            'Transaction_ID': range(1, len(df) + 1),
            'Fraud_Probability': y_pred_proba,
            'Prediction': y_pred,
            'Fraud_Label': ['FRAUD' if p == 1 else 'NORMAL' for p in y_pred],
            'Risk_Level': [self._get_risk_level(p) for p in y_pred_proba]
        })
        
        # Summary
        print("✓ Predictions complete")
        
        fraud_count = (y_pred == 1).sum()
        fraud_pct = fraud_count / len(results) * 100
        
        print(f"\n📊 Prediction Summary:")
        print(f"   Total Transactions: {len(results):,}")
        print(f"   Predicted FRAUD: {fraud_count:,} ({fraud_pct:.2f}%)")
        print(f"   Predicted NORMAL: {len(results) - fraud_count:,} ({100-fraud_pct:.2f}%)")
        
        # Risk breakdown
        print(f"\n🎯 Risk Level Breakdown:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = (results['Risk_Level'] == level).sum()
            pct = count / len(results) * 100
            print(f"   {level}: {count:,} ({pct:.2f}%)")
        
        # If we have true labels, show performance
        if has_labels:
            self._evaluate_predictions(y_true, y_pred, y_pred_proba)
        
        return results
    
    def _preprocess(self, df):
        """Preprocess transaction data"""
        print("\n⚙️  Preprocessing...")
        
        # Make copy
        df = df.copy()
        
        # Ensure all required features exist
        required_features = self.metadata['features']
        missing = [f for f in required_features if f not in df.columns]
        
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Select and order features
        df = df[required_features]
        
        # Scale Amount and Time
        if 'Amount' in df.columns:
            df['Amount'] = self.scaler.transform(df[['Amount']])
        
        print(f"✓ Preprocessed {df.shape[0]:,} transactions with {df.shape[1]} features")
        
        return df
    
    def _get_risk_level(self, probability):
        """Determine risk level"""
        if probability >= 0.9:
            return 'CRITICAL'
        elif probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _evaluate_predictions(self, y_true, y_pred, y_pred_proba):
        """Evaluate if true labels are available"""
        from sklearn.metrics import (precision_score, recall_score, f1_score, 
                                     roc_auc_score, average_precision_score,
                                     confusion_matrix)
        
        print(f"\n" + "="*70)
        print("📊 PERFORMANCE EVALUATION (True Labels Available)")
        print("="*70)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        print(f"\n✅ Model Performance:")
        print(f"   Precision:  {precision:.4f} ({precision*100:.1f}% of flagged are fraud)")
        print(f"   Recall:     {recall:.4f} ({recall*100:.1f}% of frauds caught)")
        print(f"   F1-Score:   {f1:.4f}")
        print(f"   ROC-AUC:    {roc_auc:.4f}")
        print(f"   PR-AUC:     {pr_auc:.4f} ⭐")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Positives:  {tp:,} (Frauds correctly caught)")
        print(f"   False Positives: {fp:,} (Normal flagged as fraud)")
        print(f"   True Negatives:  {tn:,} (Normal correctly identified)")
        print(f"   False Negatives: {fn:,} (Frauds missed)")
        
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\n💼 Business Metrics:")
        print(f"   Fraud Detection Rate: {fraud_detection_rate*100:.2f}%")
        print(f"   False Alarm Rate:     {false_alarm_rate*100:.2f}%")
        
        # Cost analysis
        fraud_value = 100  # Average fraud transaction value
        investigation_cost = 10  # Cost to investigate false alarm
        
        fraud_losses = fn * fraud_value
        investigation_costs = fp * investigation_cost
        frauds_prevented = tp * fraud_value
        
        print(f"\n💰 Financial Impact (Estimated):")
        print(f"   Frauds Prevented:     ${frauds_prevented:,.2f}")
        print(f"   Fraud Losses:         ${fraud_losses:,.2f}")
        print(f"   Investigation Costs:  ${investigation_costs:,.2f}")
        print(f"   Net Benefit:          ${frauds_prevented - fraud_losses - investigation_costs:,.2f}")
    
    def predict_single_transaction(self, transaction_dict):
        """
        Predict fraud for a single transaction
        
        Parameters:
        -----------
        transaction_dict : dict
            Dictionary with transaction features (Time, V1-V28, Amount)
        
        Returns:
        --------
        dict with prediction results
        """
        df = pd.DataFrame([transaction_dict])
        results = self.predict(df=df)
        
        result = results.iloc[0]
        
        return {
            'is_fraud': result['Prediction'] == 1,
            'fraud_probability': result['Fraud_Probability'],
            'risk_level': result['Risk_Level'],
            'recommendation': self._get_recommendation(result['Fraud_Probability'])
        }
    
    def _get_recommendation(self, probability):
        """Get action recommendation"""
        if probability >= 0.9:
            return "BLOCK - High confidence fraud"
        elif probability >= 0.7:
            return "REVIEW - Suspicious transaction"
        elif probability >= 0.4:
            return "MONITOR - Moderate risk"
        else:
            return "APPROVE - Low risk"

def main():
    """Demo usage"""
    print("\n" + "="*70)
    print("🚀 PROPER FRAUD DETECTION - PREDICTION SYSTEM")
    print("="*70)
    
    # Load predictor
    predictor = ProperFraudPredictor()
    if not predictor.load_model():
        return
    
    # Predict on full dataset
    print("\n" + "="*70)
    print("PREDICTING ON FULL DATASET")
    print("="*70)
    
    results = predictor.predict(data_path='creditcard.csv')
    
    # Save results
    output_file = 'fraud_predictions_proper.csv'
    results.to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved to: {output_file}")
    
    # Show samples
    print(f"\n📋 Sample Predictions:")
    print(results.head(10)[['Transaction_ID', 'Fraud_Label', 'Fraud_Probability', 'Risk_Level']].to_string(index=False))
    
    # Show high-risk transactions
    high_risk = results[results['Risk_Level'].isin(['CRITICAL', 'HIGH'])]
    if len(high_risk) > 0:
        print(f"\n⚠️  HIGH/CRITICAL RISK TRANSACTIONS: {len(high_risk):,}")
        print(high_risk.head(20)[['Transaction_ID', 'Fraud_Label', 'Fraud_Probability', 'Risk_Level']].to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ PREDICTION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
