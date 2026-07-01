# AI System for Detecting Financial Fraud in Healthcare Insurance Claims

## Problem Statement
Detect suspicious healthcare insurance claims and provider billing behavior using supervised machine learning.

## Dataset
Kaggle Healthcare Provider Fraud Detection data merged from beneficiary, inpatient, outpatient, and provider label files.

## Methodology
Claims are cleaned, date fields are normalized, inpatient and outpatient claims are combined, beneficiary attributes are joined, and provider labels are attached.
Feature engineering creates patient age, claim duration, chronic-condition counts, claim amount ratios, provider behavior metrics, beneficiary activity metrics, and code-frequency features.

## Model Results
Selected model: **LightGBM**

| model | threshold | accuracy | precision | recall | f1_score | roc_auc | pr_auc | fraud_detection_rate | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.0234 | 0.7242 | 0.5391 | 0.9523 | 0.6884 | 0.9219 | 0.8896 | 0.9523 | 0.3831 |
| XGBoost | 0.0949 | 0.7781 | 0.5979 | 0.9356 | 0.7296 | 0.9249 | 0.8922 | 0.9356 | 0.2961 |
| Random Forest | 0.2098 | 0.7465 | 0.5634 | 0.9223 | 0.6995 | 0.9069 | 0.8740 | 0.9223 | 0.3363 |
| CatBoost | 0.1864 | 0.8259 | 0.6677 | 0.9078 | 0.7694 | 0.9166 | 0.8838 | 0.9078 | 0.2126 |
| Gradient Boosting | 0.1433 | 0.7855 | 0.6111 | 0.9062 | 0.7299 | 0.9149 | 0.8841 | 0.9062 | 0.2714 |
| Tuned Random Forest | 0.2601 | 0.7736 | 0.5965 | 0.9037 | 0.7187 | 0.9112 | 0.8801 | 0.9037 | 0.2876 |
| Logistic Regression | 0.2874 | 0.8064 | 0.6501 | 0.8551 | 0.7386 | 0.9113 | 0.8760 | 0.8551 | 0.2166 |
| Decision Tree | 0.1590 | 0.7811 | 0.6314 | 0.7584 | 0.6891 | 0.7697 | 0.6482 | 0.7584 | 0.2083 |

## Final Model Metrics
- threshold: 0.023396352615010777
- accuracy: 0.7241950651560611
- precision: 0.5390587862462367
- recall: 0.9522743177046886
- f1_score: 0.688420094096221
- roc_auc: 0.9219015672882753
- pr_auc: 0.8896218538500721
- fraud_detection_rate: 0.9522743177046886
- false_positive_rate: 0.3831160279204531
- true_negative: 9368
- false_positive: 5818
- false_negative: 341
- true_positive: 6804
- model: LightGBM

## Explainability
The prediction layer returns risk level, fraud probability, global feature importance, and domain-specific local reasons such as abnormal reimbursement amount, high provider claim volume, high inpatient ratio, and unusual beneficiary activity.

## Future Improvements
- Deep learning over sequential claim histories
- Unsupervised anomaly detection for unknown fraud patterns
- Graph-based provider-beneficiary-physician fraud rings
- Real-time prediction API
- Cloud deployment and monitoring
- Scheduled model retraining with drift detection