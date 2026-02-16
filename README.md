# 🛡️ Advanced Fraud Detection System

> A production-ready, financially optimized machine learning pipeline for real-time transaction risk assessment.

This repository implements an end-to-end fraud detection solution that bridges the gap between exploratory research and high-stakes production environments. The system is engineered for **stability**, **calibration**, and **profit maximization**.

---

## 🛠️ Core Technology Stack

| Component     | Technology                        |
| :------------ | :-------------------------------- |
| **Language**  | Python 3.x                        |
| **ML Engine** | LightGBM, XGBoost                 |
| **Data Ops**  | Pandas, NumPy                     |
| **Inference** | Joblib, Isotonic Calibration      |
| **Analysis**  | Scikit-Learn, Matplotlib, Seaborn |

---

## 🚦 Key Project Highlights

- **🎯 Precision Calibration**: Uses Isotonic Regression to map model scores to true empirical fraud probabilities ($P(Fraud|Score)$).
- **💸 Financial Optimization**: Integrated an optimal decision threshold (**0.267**) mathematically tuned to minimize total business loss (Fraud Cost + Review Friction).
- **🛡️ Production Stability**: Achieved a highly stable **91.90% AUC-ROC** and **60.04% PR-AUC** with cross-validation variance $<0.3\%$.
- **⚡ Decoupled Inference**: Standalone `--predict` mode supports large-scale batch scoring without retraining.

---

## 📈 Engineering Roadmap

### 0️⃣ Data Primer

Before diving into the code, it is highly recommended to review the data dictionary and feature grouping documentation. This provides essential context on the transaction and identity attributes.

- 📖 **Documentation**: [Data Understanding](data-understanding.md)

### 1️⃣ Exploratory Foundation

Understand data distributions and established a baseline preprocessing strategy to handle high-cardinality categorical variables.

- 📖 **Walkthrough**: [Exploratory Data Analysis](eda.md)
- 💻 **Script**: `eda.py`

### 2️⃣ Stability & Performance Research

A rigorous two-phase study was conducted to select the optimal model architecture and class-balancing weights.

- 📖 **Walkthrough**: [Model Performance Analysis](model_performance_analysis.md)
- 💻 **Script**: `model_performance_analysis.py`

### 3️⃣ Production-Grade Development

Implementation of the "Gold Model" with robust column harmonization, stateful encoding, and probability calibration.

- 📖 **Walkthrough**: [Production Development Walkthrough](model_development.md)
- 💻 **Script**: `model_development.py`

### 4️⃣ Actionable Decision Layer

The system translates ML scores into clear business recommendations: `AUTO_PASS`, `MANUAL_REVIEW`, or `AUTO_BLOCK`.

---

## ⚙️ Environment Setup

Before running any scripts, ensure you have initialized the virtual environment and installed the dependencies.

```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate the environment
.\.venv\Scripts\activate

# 3. Install required libraries
pip install -r python-libraries.txt
```

---

## 🚀 Execution Guide

This project contains several scripts for different phases of the ML lifecycle. All commands should be run within the activated virtual environment.

### 1️⃣ Initial Data Discovery (EDA)
Run the exploratory analysis to generate data distribution plots and initial insights.
```powershell
.\.venv\Scripts\python eda.py
```
*Outputs: Visualizations in `images/eda-img/`*

### 2️⃣ Model Stability Research
Execute the comprehensive stability analysis to compare LightGBM vs. XGBoost and find optimal weights.
```powershell
.\.venv\Scripts\python model_performance_analysis.py
```
*Outputs: Stability plots and cost curves in `images/model-perf-analysis-img/`*

### 3️⃣ Production Training
Train the final "Gold Model" with full data, fit the probability calibrator, and save production artifacts.
```powershell
.\.venv\Scripts\python model_development.py --train
```
*Artifacts: `models/*.pkl`*

### 4️⃣ Standalone Inference
Score target data using the calibrated production model.

**Full Test Set:**
```powershell
.\.venv\Scripts\python model_development.py --predict
```

**Custom Input Batch:**
```powershell
.\.venv\Scripts\python model_development.py --predict --txn my_data_txn.csv --id my_data_id.csv --out final_results.csv --limit 1000
```

> **💡 Quick Start Note**: All scripts automatically fallback to the `ieee-fraud-detection-small/` dataset if the full `ieee-fraud-detection/` folder is not found. This allows you to run the entire pipeline immediately using the sample data included in the repository.


---

## 📂 Project Structure

```text
├── .venv/                       # Python Virtual Environment
├── .gitattributes                # Git handling for large binary CSVs
├── .gitignore                   # Excludes large data, models, and cache
├── model_development.py         # Primary Production Entry Point (Train/Predict)
├── model_development.md         # Production Implementation Details
├── model_performance_analysis.py # Stability Research & Benchmarking Script
├── model_performance_analysis.md # Detailed Performance Findings
├── eda.py                       # Preliminary Exploratory Analysis Script
├── eda.md                       # Initial Data Insights Documentation
├── data-understanding.md        # 📖 Data Dictionary & Feature Definitions
├── models/                      # Serialized Artifacts (.pkl)
│   ├── fraud_model_lgb_v1.pkl   # Trained LightGBM model
│   ├── calibrator_v1.pkl        # Isotonic probability calibrator
│   └── ...                      # Other production artifacts
├── ieee-fraud-detection/        # 📦 FULL DATASET (Ignored by Git)
│   # Download from: https://www.kaggle.com/datasets/niangmohamed/ieeecis-fraud-detection
│   ├── train_transaction.csv
│   └── ...
├── ieee-fraud-detection-small/  # 🤏 SAMPLE DATASET (Included in Git for quick start)
│   ├── train_transaction.csv
│   └── ...
├── images/                      # 📊 VISUAL ANALYTICS
│   ├── eda-img/                 # Exploratory plots (distributions, correlations)
│   └── model-perf-analysis-img/ # Stability, ROC/PR curves, & business cost curves
└── test_result.csv              # Default output for inference predictions
```
