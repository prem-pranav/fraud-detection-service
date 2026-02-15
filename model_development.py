import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set(style="whitegrid")

# Paths (relative to script location)
txn_path = "ecom-payment-txn/train_transaction.csv"
id_path = "ecom-payment-txn/train_identity.csv"
plot_dir = "images/model-development-img"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def load_and_merge_data(txn_csv, id_csv, is_test=False, nrows=None):
    """Load and merge transaction and identity datasets with column harmonization"""
    print(f"\n[1] Loading {'test' if is_test else 'train'} data (limit={nrows} rows)...")
    df_txn = pd.read_csv(txn_csv, nrows=nrows)
    
    # We load identity data. If we have a transaction limit, we should still 
    # load identity data that might match those transactions. 
    # However, for simplicity and speed in testing, we can limit identity too if needed,
    # but usually identity is smaller. Here we load full identity to ensure matches.
    df_id = pd.read_csv(id_csv) 
    
    # Harmonize Identity columns (Test uses 'id-01', Train uses 'id_01')
    if is_test:
        print("  - Harmonizing identity column names (hyphens to underscores)...")
        # Identify 'id-' columns
        id_rename_map = {col: col.replace('-', '_') for col in df_id.columns if col.startswith('id-')}
        df_id = df_id.rename(columns=id_rename_map)
    
    print(f"  - Merging on TransactionID...")
    df = pd.merge(df_txn, df_id, on='TransactionID', how='left')
    
    print(f"  - Final shape: {df.shape}")
    return df

def preprocess_features_production(df, encoders=None, is_test=False):
    """Production-grade preprocessing for tree models"""
    # 1. Prepare Target
    y = None
    if not is_test:
        y = df['isFraud'].values
    
    # 2. Drop non-predictive columns
    drop_cols = ['isFraud', 'TransactionID', 'TransactionDT']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # 3. Categorical Encoding (Label Encoding)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if encoders is None:
        encoders = {}
    
    for col in cat_cols:
        X[col] = X[col].fillna('missing').astype(str)
        
        if col not in encoders:
            if is_test:
                print(f"Warning: New categorical column {col} found in test data!")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle categories in test that weren't in training
            known_classes = set(le.classes_)
            X[col] = X[col].apply(lambda x: x if x in known_classes else 'missing')
            X[col] = le.transform(X[col])
    
    # 4. Handle Missing Values (Tree models handle -999 well)
    X = X.fillna(-999)
    
    return X, y, encoders

def compute_class_weights(y_train):
    """Compute class weight ratio"""
    print("\n" + "="*60)
    print("STEP 3: Computing Class Weights")
    print("="*60)
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    base_ratio = n_neg / n_pos
    
    print(f"\n[3.1] Class statistics:")
    print(f"Negative samples (N_neg): {n_neg}")
    print(f"Positive samples (N_pos): {n_pos}")
    print(f"Base ratio (N_neg / N_pos): {base_ratio:.4f}")
    
    weight_scales = [0.2, 0.4, 1.0, 1.5]
    weights = {f"{scale}x": base_ratio * scale for scale in weight_scales}
    
    print(f"\n[3.2] Weight configurations:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    return weights

def train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train LightGBM with class weights"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'n_jobs': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
    )
    
    return model

def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train XGBoost with class weights"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, 'validation')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    return model

def evaluate_model(model, X_val, y_val, model_name, weight_name):
    """Comprehensive evaluation with multiple metrics"""
    # Get predictions
    if 'LightGBM' in model_name:
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    else:  # XGBoost
        dval = xgb.DMatrix(X_val)
        y_pred_proba = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    # Compute metrics
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    auc_pr = average_precision_score(y_val, y_pred_proba)
    
    # Recall @ specific FPRs
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    
    # Find recall at FPR = 0.001 and 0.01
    recall_at_fpr_0001 = tpr[np.where(fpr <= 0.001)[0][-1]] if np.any(fpr <= 0.001) else 0
    recall_at_fpr_001 = tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0
    
    return {
        'Model': model_name,
        'Weight': weight_name,
        'AUC-ROC': auc_roc,
        'PR-AUC': auc_pr,
        'Recall@FPR=0.1%': recall_at_fpr_0001,
        'Recall@FPR=1%': recall_at_fpr_001,
        'predictions': y_pred_proba
    }

def plot_roc_pr_curves(results, y_val):
    """Plot ROC and PR curves for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    for result in results:
        fpr, tpr, _ = roc_curve(y_val, result['predictions'])
        label = f"{result['Model']} ({result['Weight']}): AUC={result['AUC-ROC']:.4f}"
        ax1.plot(fpr, tpr, label=label, linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    for result in results:
        precision, recall, _ = precision_recall_curve(y_val, result['predictions'])
        label = f"{result['Model']} ({result['Weight']}): AUC={result['PR-AUC']:.4f}"
        ax2.plot(recall, precision, label=label, linewidth=2)
    
    baseline = np.sum(y_val) / len(y_val)
    ax2.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.4f})', linewidth=1)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'roc_pr_curves.png'), dpi=300)
    plt.close()
    print(f"\n[Visual] ROC and PR curves saved to {plot_dir}/roc_pr_curves.png")

import argparse

def predict_fraud(txn_csv, id_csv, output_csv="test_result.csv", limit=None):
    """Inference-only pipeline using saved artifacts"""
    print("\n" + "="*60)
    print("INFERENCE MODE: GENERATING PREDICTIONS")
    if limit:
        print(f"BATCH LIMIT: {limit} rows")
    print("="*60)
    
    # 1. Load Artifacts
    print("\n[1] Loading production artifacts...")
    try:
        model = joblib.load("models/fraud_model_lgb_v1.pkl")
        encoders = joblib.load("models/feature_encoders_v1.pkl")
        calibrator = joblib.load("models/calibrator_v1.pkl")
        metadata = joblib.load("models/model_metadata_v1.pkl")
    except FileNotFoundError as e:
        print(f"Error: Missing production artifacts in models/ directory. Have you run --train?")
        return
    
    # 2. Load & Merge Data
    df = load_and_merge_data(txn_csv, id_csv, is_test=True, nrows=limit)
    
    # 3. Preprocess
    print("\n[3] Preprocessing features for inference...")
    X, _, _ = preprocess_features_production(df, encoders=encoders, is_test=True)
    
    # 4. Score & Calibrate
    print("\n[4] Scoring and Calibrating probabilities...")
    raw_scores = model.predict(X)
    calibrated_probs = calibrator.transform(raw_scores)
    
    # 5. Apply Threshold Logic
    print("\n[5] Applying decision logic...")
    threshold = metadata['optimal_threshold']
    
    results = pd.DataFrame({
        'TransactionID': df['TransactionID'],
        'isFraud_prob': calibrated_probs
    })
    
    # Add recommendation flags based on tiers
    def get_recommendation(p):
        if p >= metadata['risk_tiers']['HIGH'][0]: return 'AUTO_BLOCK'
        if p >= threshold: return 'REVIEW'
        return 'AUTO_PASS'
        
    results['recommendation'] = results['isFraud_prob'].apply(get_recommendation)
    
    # 6. Save Output
    results.to_csv(output_csv, index=False)
    print(f"\n[6] Predictions saved to: {output_csv}")
    print(f"Summary of recommendations:\n{results['recommendation'].value_counts()}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Production System")
    parser.add_argument('--train', action='store_true', help="Run production training pipeline")
    parser.add_argument('--predict', action='store_true', help="Run inference on test data")
    parser.add_argument('--txn', type=str, default="ecom-payment-txn/test_transaction.csv", help="Path to transaction CSV")
    parser.add_argument('--id', type=str, default="ecom-payment-txn/test_identity.csv", help="Path to identity CSV")
    parser.add_argument('--out', type=str, default="test_result.csv", help="Output results path")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows to process (for batch testing)")
    
    args = parser.parse_args()
    
    if args.train:
        # Re-using the previous training logic block
        print("\n" + "="*60)
        print("PRODUCTION MODEL DEVELOPMENT: LIGHTGBM + ISOTONIC CALIBRATION")
        print("="*60 + "\n")
        
        train_txn = "ecom-payment-txn/train_transaction.csv"
        train_id = "ecom-payment-txn/train_identity.csv"
        model_save_path = "models/fraud_model_lgb_v1.pkl"
        encoder_save_path = "models/feature_encoders_v1.pkl"
        calibrator_save_path = "models/calibrator_v1.pkl"
        metadata_save_path = "models/model_metadata_v1.pkl"
        
        if not os.path.exists('models'): os.makedirs('models')
        
        df = load_and_merge_data(train_txn, train_id, is_test=False)
        X, y, encoders = preprocess_features_production(df, is_test=False)
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
        
        weight_scale = 0.2
        scale_pos_weight = (np.sum(y_train == 0) / np.sum(y_train == 1)) * weight_scale
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale_pos_weight,
            'verbose': -1, 'n_jobs': -1
        }
        
        print(f"\n[4] Training Base Model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        production_model = lgb.train(params, train_data, num_boost_round=200)
        
        print("\n[5] Fitting Calibrator...")
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(production_model.predict(X_calib), y_calib)
        
        print("\n[6] Serializing Artifacts...")
        joblib.dump(production_model, model_save_path)
        joblib.dump(encoders, encoder_save_path)
        joblib.dump(calibrator, calibrator_save_path)
        joblib.dump({'optimal_threshold': 0.267, 'risk_tiers': {'LOW': (0.0, 0.15), 'MEDIUM': (0.15, 0.50), 'HIGH': (0.50, 1.0)}}, metadata_save_path)
        
        print("\nPROD TRAINING COMPLETE!")

    elif args.predict:
        predict_fraud(args.txn, args.id, args.out, limit=args.limit)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
