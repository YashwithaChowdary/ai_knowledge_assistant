# AI Data Analyst Agent – Architecture

## 1. Overall Architecture

User Question
      ↓
AI Agent
      ↓
Understand Intent
      ↓
Generate Analysis Plan
      ↓
Select Analysis Tool
      ↓
Query / Analyze Dataset
      ↓
Calculate Results with Python
      ↓
Interpret Results with AI
      ↓
Generate Insights
      ↓
User Answer

## 2. Components

### User Interface
Accepts natural-language questions from the user.

### AI Agent
Understands the question and decides what analysis should be performed.

### Analysis Planner
Creates a structured plan describing the required analysis steps.

### Data Analysis Tools
Python functions perform deterministic operations such as:

- Filtering
- Sorting
- Counting
- Aggregation
- Ranking
- Average calculations
- Percentage calculations

### Dataset
`amazon.csv`

### LLM
Interprets the user's question, selects the appropriate tool, and explains the results.

### Insight Generator
Converts calculated results into a clear natural-language explanation.

## 3. AI vs Code

| Task | Responsible Component |
|------|------------------------|
| Understand user question | AI |
| Create analysis plan | AI |
| Select analysis tool | AI |
| Filter data | Python |
| Sort data | Python |
| Calculate averages | Python |
| Count products | Python |
| Calculate rankings | Python |
| Interpret results | AI |
| Explain findings | AI |

## 4. Example

User:

"Which products have the highest ratings?"

Agent plan:

1. Validate the rating column.
2. Remove invalid or missing ratings if required.
3. Sort products by rating.
4. Select the highest-rated products.
5. Return the calculated results.
6. Explain the findings.

The numerical calculations are performed by Python.

The AI explains the results to the user.

## 5. Design Principle

The agent must not use the LLM for calculations that can be performed deterministically with Python.

AI is used for planning, reasoning, interpretation, and explanation.

Python is used for reliable data processing and calculations.