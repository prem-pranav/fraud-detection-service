# Production Model Development: Step-by-Step Walkthrough

This document tracks the transition from experimental analysis to a production-grade fraud detection system. Following the findings in the [Model Performance Analysis](model_performance_analysis.md), we have built a robust "Gold Model" optimized for financial impact and stability.

---

## Phase 3: Production Development

Following the rigorous analysis in Phase 1 and 2, we transitioned to developing the **Gold Model** for production deployment using `model_development.py`.

### Step 3.1: Robust Preprocessing Pipeline

To ensure the model performs as expected on the official test set (which has slightly different column naming conventions for device identity), we implemented a production-grade preprocessing pipeline:

1.  **Identity Harmonization**:
    - Automatically detects and renames identity columns from `id-XX` (test format) to `id_XX` (training format).
2.  **Stateful Encoding**:
    - Categorical encoders are now preserved and updated to handle "unseen" categories in production by mapping them to a safe `missing` bucket.
3.  **Data Utility**:
    - Refactored the script to support training on the **100% full dataset** to capture every available fraud pattern before final serialization.

### Step 3.2: Final Training and Serialization

The production model was built using the following finalized parameters:

- **Algorithm**: LightGBM (Gradient Boosting Decision Tree).
- **Class Weight**: 0.2x ratio (optimizing for financial profit-maximization).
- **Training Scope**: Full training dataset (590,540 rows).
- **Persistence**: The model and its associated feature encoders were serialized using `joblib` for high-speed production inference.

**Production Artifacts**:

- `models/fraud_model_lgb_v1.pkl`: The trained LightGBM model.
- `models/feature_encoders_v1.pkl`: The stateful categorical encoders.

The "Gold" model is now ready for deployment or batch inference on unlabeled test data.

### Step 3.3: Probability Calibration

To ensure the model's scores are actionable for financial risk assessment, we implemented a production-ready calibration step:

1.  **Isotonic Regression**:
    - We use a 90/10 split of the training data. The base LightGBM model is trained on 90%, and the remaining 10% is used exclusively to fit an `IsotonicRegression` wrapper.
2.  **Monotonicity & Accuracy**:
    - The calibrator maps raw boosted tree scores to true empirical probabilities, ensuring that a score of 0.8 corresponds to an actual 80% fraud likelihood.
3.  **Serialized Artifact**:
    - `models/calibrator_v1.pkl`: The saved calibrator object required for the inference pipeline.

**Business Utility**:
By providing "True Probabilities," the model allows the business team to calculate the **Expected Loss** for any transaction (Probability \* Amount), enabling precise automated decision-making.

### Step 3.4: Automated Decision Logic (Threshold-Aware)

Beyond calculating raw probabilities, the production system now includes a "Decision Layer" based on our optimized business analysis:

1.  **Optimized Threshold (0.267)**:
    - This specific value was mathematically derived to minimize the combined cost of undetected fraud and manual review friction.
2.  **Risk Tiers**:
    - **LOW (0.0 - 0.15)**: Safe regions. Recommended for auto-approval.
    - **MEDIUM (0.15 - 0.50)**: The "Optimization Range." Recommended for manual review by the fraud team.
    - **HIGH (0.50 - 1.0)**: Extreme risk. Recommended for automated blocking.
3.  **Metadata Persistence**:
    - `models/model_metadata_v1.pkl`: Stores these business rules and the versioning info. This decouples the "Decision Logic" from the hardcoded script, allowing for future threshold adjustments without retraining.

**Actionable Output**:
The model's final output now provides a clear recommendation flag for every transaction, moving from "Machine Learning" to "Business Logic."

### Step 3.5: Standalone Inference Mode

Finally, we refactored the pipeline to support decoupled execution. The system no longer requires retraining for every prediction run.

1.  **Command-Line Interface (CLI)**:
    - `--train`: Executes the full development pipeline (training + calibration + serialization).
    - `--predict`: Runs the standalone inference pipeline using pre-built artifacts.
    - `--limit <N>`: (Optional) Limits processing to the first N rows of the test dataset for rapid batch testing.
2.  **Decoupled Prediction**:
    - The `predict_fraud()` function loads the model, encoders, calibrator, and metadata independently.
    - It generates both a **Probability Score** and a **Business Recommendation** for every transaction.
3.  **Low Latency**:
    - By bypassing the training logic, the inference mode is significantly faster and uses less memory, making it suitable for large-scale production batch windows.

### Step 3.6: Interpreting Prediction Results

The inference pipeline generates a structured CSV (default: `test_result.csv`) that translates complex model probabilities into actionable business flags:

- **`TransactionID`**: Unique cross-reference to the source transaction.
- **`isFraud_prob`**: The **Calibrated Probability** (0.00 to 1.00). This represents the empirical likelihood of fraud.
- **`recommendation`**: The final decision flag based on the optimal threshold (0.267):
  - ✅ **`AUTO_PASS`**: Low risk. Transaction is cleared for processing.
  - 🔍 **`MANUAL_REVIEW`**: Medium risk (Score > 0.15). Flagged for investigation.
  - ❌ **`AUTO_BLOCK`**: High risk (Score > 0.50). Recommended for automated decline.

**Usage Example (Full Inference)**:

```powershell
.\.venv\Scripts\python model_development.py --predict --txn path/to/data.csv --out predictions.csv
```

**Usage Example (Batch Testing)**:

```powershell
.\.venv\Scripts\python model_development.py --predict --limit 100
```

---

## Appendix: Production Script

The **`model_development.py`** script handles the end-to-end production pipeline:

1. Loads and merges training data with automated column harmonization.
2. Fits and saves stateful feature encoders.
3. Trains the final model on the full training volume.
4. Serializes all artifacts to the `models/` directory.
