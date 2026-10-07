# AI Data Analyst Agent

## Project Overview

The AI Data Analyst Agent is a Week 4 Generative AI project that allows users to ask natural-language questions about an Amazon product and review dataset.

The agent combines deterministic Python data analysis with an LLM for reasoning and explanation.

## Dataset

Dataset:

`dataset/amazon.csv`

The dataset contains 1,465 product records and 16 columns related to:

- Products
- Categories
- Prices
- Discounts
- Ratings
- Rating counts
- Product descriptions
- Customer reviews

## Important Dataset Limitation

The dataset does not contain sales dates or sales quantities.

Therefore, the agent cannot perform genuine time-based sales analysis such as:

> Why did sales decrease last month?

The agent explicitly rejects unsupported questions instead of inventing an explanation.

## Architecture

```text
User Question
      |
      v
Analysis Planner
      |
      v
Select Deterministic Python Tool
      |
      v
Analyze Dataset
      |
      v
Format Results
      |
      v
AI Reasoning Layer
      |
      v
Gemini
      |
      v
Natural-Language Answer