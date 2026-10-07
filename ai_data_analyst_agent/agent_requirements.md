# AI Data Analyst Agent – Requirements

## 1. Problem

Build an AI Data Analyst Agent that allows users to ask natural-language questions about Amazon product and review data and receive data-backed insights.

## 2. Dataset

Dataset: `amazon.csv`

The dataset contains 1,465 product records and 16 columns related to products, pricing, discounts, ratings, and customer reviews.

## 3. Target User

A user who wants to understand product performance without manually writing Python or SQL queries.

## 4. Agent Responsibilities

The agent should:

1. Understand the user's question.
2. Identify the required dataset columns.
3. Create an analysis plan.
4. Select the appropriate Python data-analysis tool.
5. Perform deterministic calculations using Python.
6. Interpret the calculated results.
7. Generate a clear natural-language answer.
8. Explain the evidence behind the answer.

## 5. Supported Questions

Examples:

- Which products have the highest ratings?
- Which products have the largest discounts?
- Which categories have the most products?
- Which products have many reviews but low ratings?
- Which products offer a good combination of rating and discount?

## 6. Deterministic Code Responsibilities

Python should handle:

- Filtering
- Sorting
- Aggregation
- Counting
- Averages
- Minimum and maximum values
- Percentage calculations
- Ranking
- Data validation

## 7. AI Responsibilities

The LLM should handle:

- Understanding natural-language questions
- Creating an analysis plan
- Selecting appropriate analysis tools
- Interpreting calculated results
- Explaining findings in natural language

## 8. Important Dataset Limitation

The dataset does not contain sales dates or sales quantities.

Therefore, the agent must not claim to perform genuine time-based sales analysis such as:

"Why did sales drop last month?"

unless appropriate sales/time data is added later.

## 9. Expected Workflow

User Question
→ Understand Intent
→ Create Analysis Plan
→ Select Tool
→ Analyze Data
→ Interpret Results
→ Generate Insights

## 10. Success Criteria

The agent should:

- Produce calculations from the actual dataset.
- Avoid inventing data.
- Use Python for deterministic calculations.
- Use AI for reasoning and explanation.
- Clearly communicate limitations.
- Give reproducible answers.