# Customer Churn Prediction - Week 2

An end-to-end machine-learning project for predicting customer churn and presenting the prediction through a Streamlit interface.

## Business Problem

Which customers are likely to leave, and what action could the business take? The project uses churn probability to support prioritization of retention review.

## Dataset

Local dataset filename: `data/churn IBM dataset.csv`.

The public Telco Customer Churn dataset is associated with IBM sample data and is also distributed through public mirrors such as Kaggle. The local file used in this project is a restricted subset/export, so its results should be interpreted as results for this local data rather than automatically generalized to the complete public dataset.

Public references:
- IBM Telco customer churn sample: https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
- Kaggle Telco Customer Churn mirror: https://www.kaggle.com/blastchar/telco-customer-churn

## Project Workflow

1. Business problem definition
2. Exploratory data analysis
3. Feature engineering
4. Leakage prevention
5. Categorical encoding and numerical scaling
6. Logistic Regression
7. Random Forest
8. Model comparison
9. Business evaluation
10. Streamlit prediction interface

## Models

- Logistic Regression
- Random Forest

XGBoost was treated as optional, so the project does not add it as a mandatory dependency.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6831 | 0.6944 | 0.7511 | 0.7216 | 0.7533 |
| Random Forest | 0.6808 | 0.6980 | 0.7339 | 0.7155 | 0.7505 |

Selected model: **Logistic Regression**

Observed churn rate in the local dataset: **54.61%**.

Final model feature count: **20** before one-hot expansion.

Constant candidate features removed from modeling in this local dataset: Phone Service, Internet Service, Contract.

## Why Accuracy Is Not Enough

Churn prediction involves two types of mistakes: missing customers who actually churn and incorrectly flagging customers who stay. Precision and Recall expose those errors more clearly than Accuracy alone, while F1 balances Precision and Recall.

## Business Actions

- High risk: prioritize retention review and proactive outreach.
- Medium risk: consider targeted engagement or support follow-up.
- Low risk: continue normal service and monitoring.

These are example project actions, not approved business policy.

## Streamlit

The same Python file contains the Streamlit interface. First train the model, then launch the app.

```bash
python "customer churn.py"
streamlit run "customer churn.py"
```

## Generated Deliverables

- `outputs/day6_problem_statement.md`
- `outputs/day7_eda_notebook.ipynb`
- `outputs/day8_feature_engineering_notebook.ipynb`
- `outputs/day10_evaluation_summary.md`
- `outputs/model_comparison.csv`
- `outputs/test_predictions.csv`
- `outputs/best_churn_model.joblib`
- `outputs/model_metadata.json`
- EDA and feature-importance charts

## Important Data Limitation

The local project dataset contains only a subset of the categories present in the complete public Telco Churn dataset. In particular, some fields have limited category coverage. The generated evaluation report documents this limitation.

## GitHub

Before publishing, verify the dataset license/usage terms and do not commit secrets or personal credentials.

Suggested commands:

```bash
git init
git add .
git commit -m "Add customer churn prediction project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
