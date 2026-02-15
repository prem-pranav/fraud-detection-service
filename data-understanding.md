# Data Understanding: IEEE-CIS Fraud Detection Dataset

## Dataset Overview

The **IEEE-CIS Fraud Detection** dataset was released by the **IEEE Computational Intelligence Society (CIS)** on Kaggle. It provides a real-world benchmark for online fraud prevention.

- **Data Source**: Real-world e-commerce transactions provided by Vesta Corporation.
- **Objective**: **Binary Classification**. Predict the `isFraud` target variable:
  - `isFraud = 1`: Fraudulent transaction.
  - `isFraud = 0`: Legitimate transaction.

### Key Challenges

- **Highly Imbalanced**: Fraudulent transactions represent a tiny fraction of the total data.
- **Anonymous Features**: Many features (C, D, M, V groups) are masked or engineered, requiring deep EDA to understand their distributions.
- **Data Integration**: The dataset contains a mix of **Transaction-level** and **Identity-level** data across two separate files joined by `TransactionID`.

## Transaction Dataset: Column Groups

### A. Transaction Identifiers

| Column            | Description                                 |
| :---------------- | :------------------------------------------ |
| **TransactionID** | Unique transaction identifier               |
| **TransactionDT** | Time delta from a reference point (seconds) |

> [!WARNING]
> **TransactionDT** is relative time, not a timestamp. It is often used to create time-based features.

### B. Transaction Amount

| Column             | Description                |
| :----------------- | :------------------------- |
| **TransactionAmt** | Transaction amount (float) |

**Common analysis techniques:**

- **Log-transform**: Useful due to the wide range of values.
- **Fractional cents analysis**: Fraudulent transactions often exhibit unusual decimal patterns.

### C. Product & Card Info

| Column          | Meaning                                      |
| :-------------- | :------------------------------------------- |
| **ProductCD**   | Product category                             |
| **card1–card6** | Payment card information (hashed/anonymized) |

**Typical interpretations:**

- `card1`: Card number hash.
- `card4`: Card type (Visa, MasterCard, etc.).
- `card6`: Card category (credit/debit).

### D. Address & Location

| Column           | Meaning                                              |
| :--------------- | :--------------------------------------------------- |
| **addr1, addr2** | Billing region info (anonymized)                     |
| **dist1, dist2** | Distance metrics (e.g., shipping vs billing address) |

### E. Email Domain

| Column            | Meaning                |
| :---------------- | :--------------------- |
| **P_emaildomain** | Purchaser email domain |
| **R_emaildomain** | Recipient email domain |

> [!TIP]
> **Anonymous or rare domains** (alternative to gmail.com, yahoo.com) are often flagged as suspicious in fraud detection.

### F. Count Features (C1 - C14)

- **Purpose**: Counts and frequencies related to payment cards, addresses, and other entities.
- **Examples**: Number of addresses used, number of cards used, number of times an email address has been seen.
- **Masking**: The specific meaning of each 'C' number is hidden to protect Vesta's proprietary logic.

### G. Timedelta Features (D1 - D15)

- **Type**: Numerical.
- **Meaning**: Time differences (days) from a reference point or between events.
- **Examples**: Days since last transaction, days since account creation.
- **Significance**: These are highly important for identifying fraud patterns related to transaction "velocity".

### H. Match Features (M1 - M9) [Boolean flags]

- **Type**: Match indicators (T, F, or missing).
- **Purpose**: Indicate whether certain pieces of information match across different parts of the transaction.
- **Examples**: Name match, Address match, Card match.

### I. Vesta Engineered Features (V1 - V339)

- **Nature**: These are **Anonymized features (core mystery features)**.
- **Origin**: Derived, engineered features created by **IEEE & industry experts**.
- **Components & Logic**: These features encompass a wide range of sophisticated signals, including:
  - **Ranking**: Relative positions of transactions within certain categories.
  - **Aggregations**: Summarized behaviors (e.g., total spend over a window).
  - **Ratios**: Comparisons between different transaction attributes.
  - **Historical Behavior Signals**: Patterns derived from past activity to identify anomalies.
- **Significance**: Because they are derived from Vesta's vast historical data, they often hold significant predictive power for fraud detection, even though their raw definitions are masked.
- **Preprocessing Impact**: These columns often have high missingness (NaNs), which is why **Imputation** (Median/Mode) is critical before feeding them into a model.

## Identity Dataset: Column Groups

This dataset describes device, browser, and network information. These signals are highly predictive in many models.

### A. Device Info

| Column         | Meaning            |
| :------------- | :----------------- |
| **DeviceType** | Desktop / Mobile   |
| **DeviceInfo** | Device model or OS |

### B. Browser & OS

| Column    | Meaning           |
| :-------- | :---------------- |
| **id_31** | Browser           |
| **id_30** | OS                |
| **id_33** | Screen resolution |

### C. Network & Security

| Column          | Meaning                          |
| :-------------- | :------------------------------- |
| **id_35–id_38** | Security settings                |
| **id_01–id_11** | Identity signals (scores, flags) |
