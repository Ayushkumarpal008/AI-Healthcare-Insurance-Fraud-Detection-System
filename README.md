# AI System for Detecting Financial Fraud in Healthcare Insurance Claims

End-to-end machine learning project for detecting suspicious healthcare insurance claims using the Kaggle Healthcare Provider Fraud Detection dataset.

## Features

- Loads beneficiary, inpatient, outpatient, and provider label files.
- Merges claim records with beneficiary demographics and provider fraud labels.
- Cleans missing values, invalid dates, duplicate rows, negative amounts, and inconsistent chronic-condition flags.
- Builds claim, beneficiary, and provider behavior features.
- Handles class imbalance with class weighting, random oversampling, random undersampling, and optional SMOTE.
- Trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and optional XGBoost, LightGBM, and CatBoost models.
- Evaluates accuracy, precision, recall, F1, ROC-AUC, PR-AUC, fraud detection rate, false positive rate, and confusion matrix.
- Saves a reusable model bundle for prediction and explainability.
- Generates EDA charts, model comparison tables, and a Markdown project report.
- Provides a Streamlit dashboard for single prediction, batch CSV scoring, analytics, feature importance, suspicious providers, and prediction history.

## Folder Structure

```text
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Model_Training.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explainability.py
│   └── utils.py
├── models/
├── reports/
├── dashboards/
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Dataset Usage

The loader accepts both canonical and timestamped Kaggle filenames. The current workspace contains:

- `Train_Beneficiarydata-1542865627584.csv`
- `Train_Inpatientdata-1542865627584.csv`
- `Train_Outpatientdata-1542865627584.csv`
- `Train-1542865627584.csv`

Files may remain in the project root or be placed in `data/raw/`.

## Execution Order

1. Load, merge, clean, engineer features, train models, generate reports, and save the best model:

```bash
python -m src.train
```

2. To train on the full dataset instead of the local runtime sample:

```bash
python -m src.train --max-rows 0
```

3. Score a CSV of merged or claim-like records:

```bash
python -m src.predict --input data/processed/sample_prediction_input.csv --output data/processed/predictions.csv
```

4. Launch the dashboard:

```bash
streamlit run app/app.py
```

## Model Details

The training script compares several algorithms and selects the best recall-focused model using a threshold optimized for F2 score. Fraud detection usually values recall because missed fraud can be more costly than manual review of false positives.

The saved artifact is:

```text
models/fraud_detection_model.joblib
```

It contains the fitted cleaner, feature engineer, model preprocessor, best model, selected threshold, metrics, and explanation profile.

## Outputs

- `data/processed/train_schema.csv`
- `data/processed/train_merged_claims.csv`
- `data/processed/train_engineered_claims.csv`
- `reports/eda_summary.json`
- `reports/model_comparison.csv`
- `reports/imbalance_comparison.csv`
- `reports/training_summary.json`
- `reports/project_report.md`
- `reports/*.png` visualizations
- `models/fraud_detection_model.joblib`

## Dashboard Screenshots

Add screenshots after launching the Streamlit app:

- Single prediction view
- Batch prediction view
- Analytics view
- Feature importance view
- Suspicious providers view

## Future Enhancements

- Deep learning over patient and provider claim sequences.
- Autoencoder or isolation-forest anomaly detection.
- Graph-based fraud ring detection across providers, beneficiaries, and physicians.
- Real-time FastAPI prediction service.
- Cloud deployment with scheduled batch scoring.
- Data drift, model drift, and performance monitoring.
- Automated retraining pipeline.

