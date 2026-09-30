# Day 8 - Feature Engineering

## Engineered features

- **AvgMonthlySpend:** Total Charges divided by Tenure Months when tenure is greater than zero; otherwise Monthly Charges is used.
- **TenureGroup:** Groups tenure into 0-6, 7-12, 13-24, 25-48, and 49+ months.
- **IsNewCustomer:** Yes when Tenure Months is 6 or fewer.
- **TotalServices:** Counts service fields containing Yes.

## Categorical preprocessing

Categorical missing values are filled with the most frequent category and then one-hot encoded. Unknown categories are ignored so the application can fail safely instead of crashing.

## Numerical preprocessing

Numerical missing values use median imputation. Numerical variables are standardized for the Logistic Regression pipeline.

## Leakage prevention

The following columns are excluded from prediction: CustomerID, Churn Label, Churn Value, Churn Score, Churn Reason, CLTV. They include identifiers, the target, or fields that can encode outcome information.

## Final model features

### Categorical

- Gender
- Senior Citizen
- Partner
- Dependents
- Multiple Lines
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Paperless Billing
- Payment Method
- TenureGroup
- IsNewCustomer

### Numerical

- Tenure Months
- Monthly Charges
- Total Charges
- AvgMonthlySpend
- TotalServices

## Constant features in this local dataset

These fields had only one observed value in the supplied dataset and were excluded from the final model because they provide no variation during training:

- Phone Service
- Internet Service
- Contract
