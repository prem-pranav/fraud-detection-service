import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set(style="whitegrid")

# Paths (relative to script location)
def get_data_path(filename):
    """Helper to fallback to small dataset if main one is missing"""
    main_path = os.path.join("ieee-fraud-detection", filename)
    small_path = os.path.join("ieee-fraud-detection-small", filename)
    return main_path if os.path.exists(main_path) else small_path

txn_path = get_data_path("train_transaction.csv")
id_path = get_data_path("train_identity.csv")
plot_dir = "images/model-perf-analysis-img"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def load_and_merge_data():
    """Load and merge transaction and identity datasets"""
    print("="*60)
    print("Loading and Merging Datasets")
    print("="*60)
    
    df_txn = pd.read_csv(txn_path)
    df_id = pd.read_csv(id_path)
    df = pd.merge(df_txn, df_id, on='TransactionID', how='left')
    
    print(f"\nMerged data shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    
    return df

def preprocess_features(df):
    """Minimal preprocessing for tree models"""
    # Separate target
    y = df['isFraud'].values
    
    # Drop target and ID columns
    drop_cols = ['isFraud', 'TransactionID', 'TransactionDT']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Sort by time for proper time-based CV
    time_index = df['TransactionDT'].values if 'TransactionDT' in df.columns else np.arange(len(df))
    
    # Handle categorical columns with Label Encoding
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('missing').astype(str)
        X[col] = le.fit_transform(X[col])
    
    # Fill missing values with -999
    X = X.fillna(-999)
    
    return X.values, y, time_index, X.columns.tolist()

def train_lgb_cv(X, y, weight_scale, n_splits=5):
    """Train LightGBM with stratified cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    base_ratio = n_neg / n_pos
    scale_pos_weight = base_ratio * weight_scale
    
    fold_results = []
    all_preds = []
    all_true = []
    all_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
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
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(period=0)]
        )
        
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        auc_pr = average_precision_score(y_val, y_pred_proba)
        
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        recall_at_fpr_0001 = tpr[np.where(fpr <= 0.001)[0][-1]] if np.any(fpr <= 0.001) else 0
        recall_at_fpr_001 = tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0
        
        fold_results.append({
            'fold': fold,
            'auc_roc': auc_roc,
            'pr_auc': auc_pr,
            'recall_fpr_0001': recall_at_fpr_0001,
            'recall_fpr_001': recall_at_fpr_001
        })
        
        all_preds.extend(y_pred_proba)
        all_true.extend(y_val)
        all_importances.append(model.feature_importance(importance_type='gain'))
        
    return fold_results, np.array(all_preds), np.array(all_true), np.array(all_importances)

def train_xgb_cv(X, y, weight_scale, n_splits=5):
    """Train XGBoost with stratified cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    base_ratio = n_neg / n_pos
    scale_pos_weight = base_ratio * weight_scale
    
    fold_results = []
    all_preds = []
    all_true = []
    all_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
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
            num_boost_round=100,
            evals=[(dval, 'validation')],
            early_stopping_rounds=15,
            verbose_eval=False
        )
        
        y_pred_proba = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        auc_pr = average_precision_score(y_val, y_pred_proba)
        
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        recall_at_fpr_0001 = tpr[np.where(fpr <= 0.001)[0][-1]] if np.any(fpr <= 0.001) else 0
        recall_at_fpr_001 = tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0
        
        fold_results.append({
            'fold': fold,
            'auc_roc': auc_roc,
            'pr_auc': auc_pr,
            'recall_fpr_0001': recall_at_fpr_0001,
            'recall_fpr_001': recall_at_fpr_001
        })
        
        all_preds.extend(y_pred_proba)
        all_true.extend(y_val)
        
        # Track gain importance
        importance_dict = model.get_score(importance_type='gain')
        # Map back to array format
        importance_array = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            key = f'f{i}'
            importance_array[i] = importance_dict.get(key, 0)
        all_importances.append(importance_array)
        
    return fold_results, np.array(all_preds), np.array(all_true), np.array(all_importances)

def get_mean_metrics(fold_results):
    df_results = pd.DataFrame(fold_results)
    return {
        'auc_roc': df_results['auc_roc'].mean(),
        'pr_auc': df_results['pr_auc'].mean(),
        'auc_roc_std': df_results['auc_roc'].std()
    }

def analyze_stability(fold_results, model_name, weight_scale):
    """Analyze fold-to-fold variance"""
    df_results = pd.DataFrame(fold_results)
    
    print(f"\n{'='*60}")
    print(f"Stability Analysis: {model_name} with {weight_scale}x weight")
    print(f"{'='*60}")
    
    metrics = ['auc_roc', 'pr_auc', 'recall_fpr_0001', 'recall_fpr_001']
    stats = {}
    
    for metric in metrics:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        cv_val = (std_val / mean_val) * 100 if mean_val > 0 else 0
        
        stats[metric] = {
            'mean': mean_val,
            'std': std_val,
            'cv%': cv_val
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  CV:   {cv_val:.2f}%")
    
    return stats, df_results

def plot_fold_metrics(df_results, model_name, weight_scale):
    """Plot metrics across folds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('auc_roc', 'AUC-ROC'),
        ('pr_auc', 'PR-AUC'),
        ('recall_fpr_0001', 'Recall@FPR=0.1%'),
        ('recall_fpr_001', 'Recall@FPR=1%')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        folds = df_results['fold'].values
        values = df_results[metric].values
        mean_val = values.mean()
        std_val = values.std()
        
        ax.plot(folds, values, marker='o', linewidth=2, markersize=8, label='Per-fold score')
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.fill_between(folds, mean_val - std_val, mean_val + std_val, alpha=0.2, color='r')
        
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} Across Folds ({model_name} {weight_scale}x)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fold_metrics_{model_name}_{weight_scale}x.png'), dpi=300)
    plt.close()
    print(f"[Visual] Fold metrics plot saved for {model_name} {weight_scale}x")

def plot_roc_pr_cv(all_true, all_preds, fold_results, model_name, weight_scale):
    """Plot ROC and PR curves with CV confidence bands"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_true, all_preds)
    auc_mean = np.mean([r['auc_roc'] for r in fold_results])
    auc_std = np.std([r['auc_roc'] for r in fold_results])
    
    ax1.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC={auc_mean:.4f}±{auc_std:.4f})', color='blue')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curve ({model_name} {weight_scale}x)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(all_true, all_preds)
    pr_mean = np.mean([r['pr_auc'] for r in fold_results])
    pr_std = np.std([r['pr_auc'] for r in fold_results])
    baseline = np.sum(all_true) / len(all_true)
    
    ax2.plot(recall, precision, linewidth=3, label=f'PR (AUC={pr_mean:.4f}±{pr_std:.4f})', color='green')
    ax2.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.4f})', linewidth=1)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'Precision-Recall Curve ({model_name} {weight_scale}x)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'roc_pr_{model_name}_{weight_scale}x.png'), dpi=300)
    plt.close()
    print(f"[Visual] ROC and PR curves saved for {model_name} {weight_scale}x")

def plot_score_distributions(all_true, all_preds, model_name, weight_scale):
    """Plot score distributions for fraud vs non-fraud"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    fraud_scores = all_preds[all_true == 1]
    non_fraud_scores = all_preds[all_true == 0]
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(non_fraud_scores, bins=50, alpha=0.6, label='Non-Fraud', color='blue', density=True)
    ax1.hist(fraud_scores, bins=50, alpha=0.6, label='Fraud', color='red', density=True)
    ax1.set_xlabel('Predicted Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Score Distribution ({model_name} {weight_scale}x)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [non_fraud_scores, fraud_scores]
    bp = ax2.boxplot(data_to_plot, labels=['Non-Fraud', 'Fraud'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax2.set_ylabel('Predicted Score', fontsize=12)
    ax2.set_title(f'Score Distribution Boxplot ({model_name} {weight_scale}x)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'score_dist_{model_name}_{weight_scale}x.png'), dpi=300)
    plt.close()
    print(f"[Visual] Score distribution plots saved for {model_name} {weight_scale}x")

def plot_calibration_curve(all_true, all_preds, model_name, weight_scale):
    """Plot reliability diagram to check probability calibration"""
    prob_true, prob_pred = calibration_curve(all_true, all_preds, n_bins=10)
    
    plt.figure(figsize=(10, 7))
    plt.plot(prob_pred, prob_true, marker='s', linewidth=2, label=f'{model_name} ({weight_scale}x)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Reliability Diagram: {model_name} {weight_scale}x', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'calibration_{model_name}_{weight_scale}x.png'), dpi=300)
    plt.close()
    print(f"[Visual] Calibration plot saved for {model_name} {weight_scale}x")

def plot_feature_importance_stability(all_importances, feature_names, model_name, weight_scale, top_n=20):
    """Plot feature importance and its stability across folds"""
    imp_df = pd.DataFrame(all_importances, columns=feature_names)
    
    # Calculate mean and std
    mean_imp = imp_df.mean().sort_values(ascending=False).head(top_n)
    std_imp = imp_df[mean_imp.index].std()
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=mean_imp.values, y=mean_imp.index, color='skyblue', label='Mean Gain')
    
    # Add error bars manually for stability visualization
    plt.errorbar(x=mean_imp.values, y=np.arange(top_n), xerr=std_imp.values, fmt='none', c='black', capsize=3)
    
    plt.xlabel('Mean Feature Gain (Importance)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance Stability ({model_name} {weight_scale}x)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'feat_imp_stability_{model_name}_{weight_scale}x.png'), dpi=300)
    plt.close()
    
    # Log top 5 for terminal output
    print(f"\nTop 5 Most Stable Drivers (Weight: {weight_scale}x):")
    for feat in mean_imp.index[:5]:
        cv = (imp_df[feat].std() / imp_df[feat].mean()) * 100
        print(f"  - {feat}: CV = {cv:.2f}%")
    
    print(f"[Visual] Feature importance stability plot saved for {model_name} {weight_scale}x")

def plot_cost_optimization(all_true, all_preds, model_name, weight_scale):
    """Identify the optimal threshold by minimizing business cost"""
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # Business Assumptions (Customizable)
    cost_fp = 10.0   # Cost of a manual review/customer friction (USD)
    cost_fn = 150.0  # Average loss from an undetected fraud (USD)
    
    costs = []
    for t in thresholds:
        y_pred = (all_preds >= t).astype(int)
        fp = np.sum((y_pred == 1) & (all_true == 0))
        fn = np.sum((y_pred == 0) & (all_true == 1))
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(total_cost)
    
    min_cost = min(costs)
    opt_threshold = thresholds[np.argmin(costs)]
    
    # Baseline cost (do nothing - assume all fraud is missed)
    baseline_cost = np.sum(all_true == 1) * cost_fn
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, linewidth=3, label='Business Cost Curve')
    plt.axvline(x=opt_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {opt_threshold:.3f}')
    plt.axhline(y=baseline_cost, color='k', linestyle=':', label='Baseline (No Detection)')
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Total Cost (USD)', fontsize=12)
    plt.title(f'Cost-Sensitive Threshold Optimization ({model_name} {weight_scale}x)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotation for savings
    savings = baseline_cost - min_cost
    plt.annotate(f'Max Savings: ${savings:,.0f}', 
                 xy=(opt_threshold, min_cost), 
                 xytext=(opt_threshold + 0.1, min_cost + (baseline_cost-min_cost)*0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'cost_optimization_{model_name}_{weight_scale}x.png'), dpi=300)
    plt.close()
    
    print(f"\nCost Optimization Results (Weight: {weight_scale}x):")
    print(f"  - Optimal Threshold: {opt_threshold:.3f}")
    print(f"  - Minimum Total Cost: ${min_cost:,.2f}")
    print(f"  - Estimated Savings vs Baseline: ${savings:,.2f}")
    print(f"[Visual] Cost optimization plot saved for {model_name} {weight_scale}x")

def main():
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60 + "\n")
    
    # Load full data
    df_full = load_and_merge_data()
    X_full, y_full, _, feat_names = preprocess_features(df_full)
    
    # 1. SCREENING PHASE (using a sample for speed)
    print("\n" + "#"*60)
    print("PHASE 1: SCREENING MODELS & WEIGHT RATIOS (Sample size: 100,000)")
    print("#"*60 + "\n")
    
    # Sample for screening
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_full), min(100000, len(y_full)), replace=False)
    X_sample, y_sample = X_full[sample_indices], y_full[sample_indices]
    
    weight_ratios = [0.2, 0.4, 0.6, 1.0, 1.5, 2.0]
    models_to_test = {
        'LightGBM': train_lgb_cv,
        'XGBoost': train_xgb_cv
    }
    
    best_results = []
    
    for model_name, train_func in models_to_test.items():
        print(f"Screening {model_name}...")
        for ratio in weight_ratios:
            try:
                # Use fewer folds (3) for faster screening
                fold_results, _, _, _ = train_func(X_sample, y_sample, ratio, n_splits=3)
                metrics = get_mean_metrics(fold_results)
                best_results.append({
                    'Model': model_name,
                    'Ratio': ratio,
                    'AUC-ROC': metrics['auc_roc'],
                    'PR-AUC': metrics['pr_auc'],
                    'AUC-Std': metrics['auc_roc_std']
                })
                print(f"  Ratio {ratio}x: AUC-ROC = {metrics['auc_roc']:.4f}, PR-AUC = {metrics['pr_auc']:.4f}")
            except Exception as e:
                print(f"  Error screening {model_name} at ratio {ratio}: {str(e)}")
            
    screening_df = pd.DataFrame(best_results)
    screening_df = screening_df.sort_values(by='AUC-ROC', ascending=False)
    
    print("\n--- SCREENING RESULTS SUMMARY ---")
    print(screening_df.to_string(index=False))
    
    # Identify top model and its top 2 weights
    if len(screening_df) == 0:
        print("Error: No models were successfully screened.")
        return
        
    top_model_name = screening_df.iloc[0]['Model']
    top_two_ratios = screening_df[screening_df['Model'] == top_model_name].head(2)['Ratio'].tolist()
    
    print("\n" + "#"*60)
    print(f"PHASE 2: DETAILED ANALYSIS FOR {top_model_name.upper()} (Full Dataset)")
    print(f"Top 2 ratios: {top_two_ratios[0]}x and {top_two_ratios[1]}x")
    print("#"*60 + "\n")
    
    # 2. DETAILED ANALYSIS PHASE (using full data)
    detailed_results = {}
    train_func = models_to_test[top_model_name]
    
    for weight_scale in top_two_ratios:
        # Run CV again for full prediction data (needed for plots)
        fold_results, all_preds, all_true, all_importances = train_func(X_full, y_full, weight_scale, n_splits=5)
        
        # Analyze stability
        stats, df_results = analyze_stability(fold_results, top_model_name, weight_scale)
        
        # Generate visualizations
        plot_fold_metrics(df_results, top_model_name, weight_scale)
        plot_roc_pr_cv(all_true, all_preds, fold_results, top_model_name, weight_scale)
        plot_score_distributions(all_true, all_preds, top_model_name, weight_scale)
        plot_calibration_curve(all_true, all_preds, top_model_name, weight_scale)
        plot_feature_importance_stability(all_importances, feat_names, top_model_name, weight_scale)
        plot_cost_optimization(all_true, all_preds, top_model_name, weight_scale)
        
        detailed_results[f"{weight_scale}x"] = {
            'stats': stats,
            'fold_results': df_results
        }
    
    # Final comparison
    print("\n" + "="*60)
    print(f"FINAL PERFORMANCE ANALYSIS: {top_model_name}")
    print("="*60 + "\n")
    
    comparison_data = []
    for weight, results in detailed_results.items():
        stats = results['stats']
        comparison_data.append({
            'Weight': weight,
            'AUC-ROC (mean±std)': f"{stats['auc_roc']['mean']:.4f}±{stats['auc_roc']['std']:.4f}",
            'AUC-ROC CV%': f"{stats['auc_roc']['cv%']:.2f}%",
            'PR-AUC (mean±std)': f"{stats['pr_auc']['mean']:.4f}±{stats['pr_auc']['std']:.4f}",
            'PR-AUC CV%': f"{stats['pr_auc']['cv%']:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print(f"Detailed Analysis of {top_model_name} Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
