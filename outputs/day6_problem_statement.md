# Day 6 - Customer Churn Problem Statement

## Business Problem

The goal of this project is to predict which customers are likely to leave the service. The prediction can help a business prioritize customers for retention review before deciding what action, if any, is appropriate.

## Potential Business Value

A churn prediction system can help the business focus limited retention resources on customers who show higher estimated churn risk. This can support earlier customer-service review, targeted communication, service education, or other retention programs. The model should support human review rather than automatically decide customer treatment.

## Possible Actions

- **High risk:** prioritize a retention review and consider proactive outreach.
- **Medium risk:** consider proactive engagement, service/support follow-up, or targeted communication.
- **Low risk:** continue normal service and monitoring.

These are example actions for the academic project and should not be treated as approved business policy.

## Dataset Context

The local project file contains 2,128 customer records after cleaning, with an observed churn rate of 54.61%.

## Target

`Churn Label` is converted into a binary target called `Churn`, where Yes = 1 and No = 0.

## Success Measures

The project compares Logistic Regression and Random Forest using Accuracy, Precision, Recall, F1 Score, and ROC-AUC. F1 Score is used as the primary model-selection metric with Recall as the tie-breaker because the project needs to balance missed churners against false alarms.
