# Day 10 - Model Evaluation and Business Value

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6831 | 0.6944 | 0.7511 | 0.7216 | 0.7533 |
| Random Forest | 0.6808 | 0.6980 | 0.7339 | 0.7155 | 0.7505 |

## Selected Model

**Logistic Regression** was selected because it has the highest F1 Score in this run, with Recall used as the tie-breaker.

## Why Accuracy Alone Is Not Enough

Accuracy measures the percentage of all test customers classified correctly. In a churn project, however, the business also needs to understand how many actual churners were found and how many customers were flagged incorrectly. Precision measures the proportion of predicted churners who actually churned; Recall measures the proportion of actual churners that the model detected; F1 combines Precision and Recall. Therefore, looking only at Accuracy could hide an important business trade-off.

For context, the majority-class baseline accuracy in this dataset is approximately 0.5461. The selected model's test Accuracy is 0.6831.

## Selected Model Test Confusion Matrix

- True Negatives: 116
- False Positives: 77
- False Negatives: 58
- True Positives: 175

False negatives represent customers who churned but were classified as non-churners, which can mean missed retention opportunities. False positives represent customers who were flagged as churn risks but did not churn, which can consume retention resources unnecessarily. The appropriate balance depends on business costs and capacity.

## Example Business Action

Use the model to prioritize review rather than to automatically make customer decisions. High-risk cases can be reviewed first for appropriate outreach, service issues, or other approved retention options. Medium-risk cases can receive lower-intensity follow-up, while low-risk cases can remain in normal monitoring.

## Risk Categories Used by the Demo App

- Low: churn probability < 30%
- Medium: 30% to less than 60%
- High: churn probability >= 60%

These thresholds are initial demonstration thresholds, not validated business thresholds.

## Limitations

The local dataset is not the complete original public Telco Customer Churn table. The current project file contains a restricted subset, including limited category coverage for some variables. Model results therefore describe this local dataset and should not be generalized automatically to a different population.
