import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import os

# Set aesthetic style
sns.set(style="whitegrid")

# Paths (relative to script location)
txn_path = "ecom-payment-txn/train_transaction.csv"
id_path = "ecom-payment-txn/train_identity.csv"
plot_dir = "images/eda-img"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def perform_detailed_eda(txn_path, id_path, sample_n=50000):
    print("="*60)
    print("--- Detailed EDA: Merged Transaction & Identity ---")
    print("="*60)

    print("\n" + "#"*60)
    print("### ADVANCED FEATURE ANALYSIS & INSIGHTS ###")
    print("#"*60 + "\n")

    # 1. Merged Data EDA Sanity Checks
    print(f"\n[1] Loading {sample_n} rows of transaction data...")
    df_txn = pd.read_csv(txn_path, nrows=sample_n)
    print(f"[1] Loading full identity data...")
    df_id = pd.read_csv(id_path)
    
    print("[1] Merging datasets on TransactionID...")
    df = pd.merge(df_txn, df_id, on='TransactionID', how='left')
    print(f"Merged Shape: {df.shape}")

    # 1.1 Class Imbalance
    print("\n" + "="*60)
    print("[1.1] Class Imbalance Check")
    print("="*60)
    fraud_counts = df['isFraud'].value_counts()
    fraud_pct = df['isFraud'].value_counts(normalize=True) * 100
    print(f"Counts:\n{fraud_counts}")
    print(f"\nPercentages:\n{fraud_pct.map(lambda x: f'{x:.2f}%').to_string()}")

    print("\n[B] Generating Bar Plot of isFraud...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='isFraud', data=df, hue='isFraud', palette='viridis', legend=False)
    plt.title('Distribution of isFraud (Target Variable)')
    plt.xlabel('isFraud (0: Legitimate, 1: Fraud)')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.savefig(os.path.join(plot_dir, "fraud_distribution.png"))
    plt.close()

    # 1.2 Cross-Validation: Fraud Ratio Stability
    print("\n" + "="*60)
    print("[1.2] Cross-Validation: Fraud Ratio Stability")
    print("="*60)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = df['isFraud'].values
    fold_ratios = []
    print(f"{'Fold':<10} | {'Train Ratio':<15} | {'Val Ratio':<15}")
    print("-" * 45)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        train_ratio = np.mean(y[train_idx]) * 100
        val_ratio = np.mean(y[val_idx]) * 100
        print(f"{fold:<10} | {train_ratio:>13.4f}% | {val_ratio:>13.4f}%")
        fold_ratios.append(val_ratio)
    print("-" * 45)
    print(f"Average Validation Fraud Ratio: {np.mean(fold_ratios):.4f}%")
    print(f"Standard Deviation of Ratios:  {np.std(fold_ratios):.4f}%")

    # 2. Transaction Amount Analysis
    print("\n" + "="*60)
    print("[2] Transaction Amount Analysis")
    print("="*60)
    print("\n[A] Generating TransactionAmt Distribution Histogram (Log Scale)...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TransactionAmt'], bins=50, kde=True, log_scale=True, color='blue')
    plt.title('Distribution of TransactionAmt (Log Scale)')
    plt.savefig(os.path.join(plot_dir, 'transaction_amt_dist.png'))
    plt.close()

    print("\n[B] Generating TransactionAmt vs isFraud Boxplot...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='isFraud', y='TransactionAmt', data=df, hue='isFraud', palette='Set2', legend=False)
    plt.yscale('log')
    plt.title('TransactionAmt vs isFraud (Log Scale)')
    plt.savefig(os.path.join(plot_dir, 'transaction_amt_vs_fraud.png'))
    plt.close()

    print("\n[C] Computing Fraud Rate per Amount Bin...")
    df['AmtBin'] = pd.qcut(df['TransactionAmt'], q=10, duplicates='drop')
    bin_fraud_rate = df.groupby('AmtBin', observed=True)['isFraud'].mean() * 100
    print("\nFraud Rate per TransactionAmt Decile:")
    print(bin_fraud_rate.map(lambda x: f'{x:.2f}%').to_string())
    plt.figure(figsize=(12, 6))
    bin_fraud_rate.plot(kind='bar', color='salmon')
    plt.title('Fraud Rate by TransactionAmt Deciles')
    plt.savefig(os.path.join(plot_dir, 'fraud_rate_by_amt_bin.png'))
    plt.close()

    # 3. Product & Payment Behavior Analysis
    print("\n" + "="*60)
    print("[3] Product & Payment Behavior Analysis")
    print("="*60)
    print("\n[A] Computing Fraud Rate per ProductCD...")
    product_fraud_rate = df.groupby('ProductCD')['isFraud'].mean() * 100
    print(product_fraud_rate.map(lambda x: f'{x:.2f}%').to_string())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=product_fraud_rate.index, y=product_fraud_rate.values, hue=product_fraud_rate.index, palette='viridis', legend=False)
    plt.title('Fraud Rate by ProductCD')
    plt.savefig(os.path.join(plot_dir, 'fraud_rate_by_product.png'))
    plt.close()

    print("\n[B] Analyzing Card Features (card1-card6)...")
    for col in ['card4', 'card6']:
        if col in df.columns:
            print(f"\nFraud Rate per {col}:")
            rate = df.groupby(col)['isFraud'].mean() * 100
            print(rate.map(lambda x: f'{x:.2f}%').to_string())
            plt.figure(figsize=(10, 6))
            sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette='magma', legend=False)
            plt.title(f'Fraud Rate by {col}')
            plt.savefig(os.path.join(plot_dir, f'fraud_rate_by_{col}.png'))
            plt.close()

    # 4. Time-based Analysis
    print("\n" + "="*60)
    print("[4] Time-based Analysis")
    print("="*60)
    df['Hour'] = (df['TransactionDT'] / 3600) % 24
    df['Day'] = df['TransactionDT'] / (3600 * 24)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Day'], bins=50, color='skyblue', kde=True)
    plt.title('Transaction Volume over Time (Days)')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'transaction_volume_over_time.png'))
    plt.close()

    df['DayBin'] = pd.cut(df['Day'], bins=20)
    day_fraud_rate = df.groupby('DayBin', observed=True)['isFraud'].mean() * 100
    plt.figure(figsize=(12, 6))
    day_fraud_rate.plot(kind='line', marker='o', color='red')
    plt.title('Fraud Rate over Time (Days)')
    plt.xlabel('Time Buckets (Days)')
    plt.ylabel('Fraud Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'fraud_rate_over_time.png'))
    plt.close()

    df['HourBin'] = df['Hour'].astype(int)
    hour_fraud_rate = df.groupby('HourBin', observed=True)['isFraud'].mean() * 100
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hour_fraud_rate.index, y=hour_fraud_rate.values, hue=hour_fraud_rate.index, palette='coolwarm', legend=False)
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate (%)')
    plt.savefig(os.path.join(plot_dir, 'fraud_rate_by_hour.png'))
    plt.close()

    # 5. Email Domain Analysis
    print("\n" + "="*60)
    print("[5] Email Domain Analysis")
    print("="*60)
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            print(f"\n[A] Top 10 frequencies for {col}:")
            top_domains = df[col].value_counts().head(10)
            print(top_domains)
            fraud_rate = df[df[col].isin(top_domains.index)].groupby(col)['isFraud'].mean() * 100
            print(f"Fraud rate for top 10 {col}s:")
            print(fraud_rate.map(lambda x: f'{x:.2f}%').sort_values(ascending=False).to_string())
            plt.figure(figsize=(12, 6))
            sns.barplot(x=fraud_rate.index, y=fraud_rate.values, hue=fraud_rate.index, palette='viridis', legend=False)
            plt.title(f'Fraud Rate by Top 10 {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'fraud_rate_by_{col}.png'))
            plt.close()

    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        print("\n[C] Comparing P_emaildomain vs R_emaildomain...")
        df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
        match_fraud_rate = df.groupby('email_match')['isFraud'].mean() * 100
        print(f"Fraud Rate by Email Match:\n{match_fraud_rate.map(lambda x: f'{x:.2f}%')}")
        plt.figure(figsize=(8, 5))
        sns.barplot(x=match_fraud_rate.index, y=match_fraud_rate.values, hue=match_fraud_rate.index, palette='Set1', legend=False)
        plt.title('Fraud Rate: Email Domain Match vs No Match')
        plt.savefig(os.path.join(plot_dir, 'fraud_rate_by_email_match.png'))
        plt.close()

    # 6. Address & Distance Analysis
    print("\n" + "="*60)
    print("[6] Address & Distance Analysis")
    print("="*60)
    for col in ['addr1', 'addr2']:
        if col in df.columns:
            print(f"\n[A] Top 10 frequencies and fraud rate for {col}:")
            top_addr = df[col].value_counts().head(10).index
            addr_fraud = df[df[col].isin(top_addr)].groupby(col)['isFraud'].mean() * 100
            print(addr_fraud.map(lambda x: f'{x:.2f}%').to_string())
            plt.figure(figsize=(10, 5))
            sns.barplot(x=addr_fraud.index, y=addr_fraud.values, hue=addr_fraud.index, palette='coolwarm', legend=False)
            plt.title(f'Fraud Rate by Top 10 {col}')
            plt.savefig(os.path.join(plot_dir, f'fraud_rate_by_{col}.png'))
            plt.close()

    for col in ['dist1', 'dist2']:
        if col in df.columns:
            print(f"\n[B] Generating distribution for {col} by isFraud...")
            plot_df = df[df[col] > 0][[col, 'isFraud']].dropna()
            if not plot_df.empty and plot_df[col].nunique() > 1:
                plt.figure(figsize=(10, 5))
                try:
                    sns.histplot(data=plot_df, x=col, hue='isFraud', kde=True, element="step", common_norm=False, log_scale=True)
                    plt.title(f'Distribution of {col} (Log Scale) by Fraud Label')
                except:
                    plt.clf()
                    sns.histplot(data=plot_df, x=col, hue='isFraud', kde=False, element="step", common_norm=False, log_scale=True)
                    plt.title(f'Distribution of {col} (Log Scale) by Fraud Label')
                plt.savefig(os.path.join(plot_dir, f'dist_labels_{col}.png'))
                plt.close()

    # 7. Match Features (M1–M9)
    print("\n" + "="*60)
    print("[7] Match Features (M1–M9)")
    print("="*60)
    m_cols = [f'M{i}' for i in range(1, 10)]
    print("\n[A] Missing rate per M-column:")
    print((df[m_cols].isnull().mean() * 100).map(lambda x: f'{x:.2f}%'))
    for col in m_cols:
        if col in df.columns:
            temp_df = df[[col, 'isFraud']].copy()
            temp_df[col] = temp_df[col].fillna('Missing')
            m_fraud = temp_df.groupby(col)['isFraud'].mean() * 100
            print(f"\n{col} Fraud Rate:\n{m_fraud.map(lambda x: f'{x:.2f}%')}")
            plt.figure(figsize=(8, 5))
            sns.barplot(x=m_fraud.index, y=m_fraud.values, hue=m_fraud.index, palette='viridis', legend=False)
            plt.title(f'Fraud Rate for {col}')
            plt.savefig(os.path.join(plot_dir, f'fraud_rate_{col}.png'))
            plt.close()

    # 8. Count & Behavior Features (C1–C14)
    print("\n" + "="*60)
    print("[8] Count & Behavior Features (C-series)")
    print("="*60)
    c_cols = [f'C{i}' for i in range(1, 15)]
    if all(col in df.columns for col in c_cols):
        print("\n[A] Generating boxplots for C-series features...")
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(c_cols[:6], 1):
            plt.subplot(2, 3, i)
            sns.boxplot(x='isFraud', y=col, data=df, hue='isFraud', palette='Set1', legend=False)
            plt.yscale('log')
            plt.title(f'{col} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'c_series_distribution.png'))
        plt.close()

    # 9. Time-delta Features (D1–D15)
    print("\n" + "="*60)
    print("[9] Time-delta Features (D-series)")
    print("="*60)
    for col in ['D1', 'D4']:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.kdeplot(data=df, x=col, hue='isFraud', common_norm=False)
            plt.title(f'{col} Distribution: Days Since Event')
            plt.savefig(os.path.join(plot_dir, f'distribution_{col}.png'))
            plt.close()

    # 10. Identity Data Analysis
    print("\n" + "="*60)
    print("[10] Identity Data Analysis")
    print("="*60)
    if 'DeviceType' in df.columns:
        device_fraud = df.groupby('DeviceType')['isFraud'].mean() * 100
        print(f"\nFraud Rate by DeviceType:\n{device_fraud.map(lambda x: f'{x:.2f}%')}")
        plt.figure(figsize=(8, 5))
        sns.barplot(x=device_fraud.index, y=device_fraud.values, hue=device_fraud.index, palette='plasma', legend=False)
        plt.title('Fraud Rate by DeviceType')
        plt.savefig(os.path.join(plot_dir, 'fraud_rate_by_devicetype.png'))
        plt.close()

    for col in ['id_31', 'id_30']:
        if col in df.columns:
            counts = df[col].value_counts()
            frequent_cats = counts[counts > 50].index
            rate = df[df[col].isin(frequent_cats)].groupby(col)['isFraud'].mean() * 100
            top_rate = rate.sort_values(ascending=False).head(10)
            print(f"\nTop 10 Fraud Rates by {col}:\n{top_rate.map(lambda x: f'{x:.2f}%')}")
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_rate.index, y=top_rate.values, hue=top_rate.index, palette='viridis', legend=False)
            plt.title(f'Top 10 Fraud Rates by {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'fraud_rate_by_{col}.png'))
            plt.close()

    # 11. Statistical & Summary Metrics
    print("\n" + "="*60)
    print("[11] Statistical & Summary Metrics")
    print("="*60)
    missing_rates = df.isnull().mean() * 100
    print("\n[11.1] Missing Value Analysis (Top 20):")
    print(missing_rates.sort_values(ascending=False).head(20).map(lambda x: f'{x:.2f}%'))

    print("\n[11.2] Correlation & Redundancy:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_numeric = missing_rates[missing_rates < 50].index.intersection(numeric_cols)
    correlations = df[valid_numeric].corr()['isFraud'].sort_values(ascending=False)
    print(f"Top 10 Positively Correlated:\n{correlations.head(11)}")
    plt.figure(figsize=(12, 10))
    top_corr_features = correlations.abs().sort_values(ascending=False).head(20).index
    sns.heatmap(df[top_corr_features].corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap (Top 20 Features)')
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'))
    plt.close()

    print("\n[11.3] High-cardinality Categorical Risk (card1):")
    if 'card1' in df.columns:
        card1_counts = df['card1'].value_counts()
        card_risk = df.groupby('card1')['isFraud'].mean() * 100
        plt.figure(figsize=(10, 6))
        plt.scatter(np.log1p(card1_counts), card_risk, alpha=0.5, color='purple')
        plt.title('Log(Count) vs Fraud Rate (card1)')
        plt.xlabel('Log(Count)')
        plt.ylabel('Fraud Rate (%)')
        plt.savefig(os.path.join(plot_dir, 'card1_risk_pattern.png'))
        plt.close()

    print("\n[11.4] Leakage Checks (Time-split):")
    split_idx = int(len(df) * 0.7)
    print(f"Train Fraud Rate: {df.iloc[:split_idx]['isFraud'].mean()*100:.2f}%")
    print(f"Validation Fraud Rate: {df.iloc[split_idx:]['isFraud'].mean()*100:.2f}%")

    print("\n[11.5] EDA Summary Metrics Table:")
    summary_data = []
    interesting_features = list(set(['TransactionAmt', 'ProductCD', 'card1', 'card4', 'addr1', 'dist1'] + list(top_corr_features[:10])))
    for feat in interesting_features:
        if feat in df.columns:
            summary_data.append({
                'Feature': feat,
                'Missing %': f'{df[feat].isnull().mean()*100:.2f}%',
                'Unique': df[feat].nunique(),
                'Fraud (P)%': f'{df[df[feat].notnull()]["isFraud"].mean()*100:.2f}%',
                'Fraud (M)%': f'{df[df[feat].isnull()]["isFraud"].mean()*100:.2f}%' if df[feat].isnull().any() else 'N/A'
            })
    print(pd.DataFrame(summary_data).to_string(index=False))

    print("\n" + "="*60)
    print("EDA Sanity Checks Complete.")
    print("="*60)

if __name__ == "__main__":
    if os.path.exists(txn_path) and os.path.exists(id_path):
        perform_detailed_eda(txn_path, id_path)
    else:
        print("Error: Dataset files not found. Please check paths:")
        print(f"TXN: {txn_path}")
        print(f"ID:  {id_path}")
