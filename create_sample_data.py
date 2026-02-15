import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_samples():
    input_dir = 'ecom-payment-txn'
    output_dir = 'data-sample'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("--- Creating Sampled Dataset (Stratified) ---")
    
    # 1. Train Transactions (Stratified)
    print("Processing train_transaction.csv...")
    df_train_txn = pd.read_csv(f'{input_dir}/train_transaction.csv')
    sample_train_txn, _ = train_test_split(
        df_train_txn, 
        train_size=50000, 
        stratify=df_train_txn['isFraud'], 
        random_state=42
    )
    sample_train_txn.to_csv(f'{output_dir}/train_transaction.csv', index=False)
    print(f"  - Saved {len(sample_train_txn)} rows.")

    # 2. Train Identity
    print("Processing train_identity.csv...")
    df_train_id = pd.read_csv(f'{input_dir}/train_identity.csv')
    sample_train_id = df_train_id[df_train_id['TransactionID'].isin(sample_train_txn['TransactionID'])]
    sample_train_id.to_csv(f'{output_dir}/train_identity.csv', index=False)
    print(f"  - Saved {len(sample_train_id)} rows.")

    # 3. Test Transactions (Random)
    print("Processing test_transaction.csv...")
    df_test_txn = pd.read_csv(f'{input_dir}/test_transaction.csv')
    sample_test_txn = df_test_txn.sample(n=20000, random_state=42)
    sample_test_txn.to_csv(f'{output_dir}/test_transaction.csv', index=False)
    print(f"  - Saved {len(sample_test_txn)} rows.")

    # 4. Test Identity
    print("Processing test_identity.csv...")
    df_test_id = pd.read_csv(f'{input_dir}/test_identity.csv')
    sample_test_id = df_test_id[df_test_id['TransactionID'].isin(sample_test_txn['TransactionID'])]
    sample_test_id.to_csv(f'{output_dir}/test_identity.csv', index=False)
    print(f"  - Saved {len(sample_test_id)} rows.")

    print(f"\nSuccess! Sampled dataset created in '{output_dir}/'")

if __name__ == "__main__":
    create_samples()
