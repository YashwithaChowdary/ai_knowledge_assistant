# RAG Answer Quality Evaluation

## Purpose

Evaluate whether the AI Knowledge Assistant generates answers that are grounded in the provided documents.

## Evaluation criteria

For each question, check:

1. Relevance - Does the answer directly address the question?
2. Grounding - Is the answer supported by the retrieved document evidence?
3. Correctness - Does the answer accurately reflect the source material?
4. Completeness - Does it include the important points needed to answer the question?
5. Hallucination avoidance - Does it avoid unsupported information?
6. Source references - Are the cited documents relevant to the answer?

## Questions

1. What is decomposition in computational thinking?
2. What is abstraction in computer science?
3. What is recursion?
4. What is a base case in recursion?
5. What is the purpose of testing and debugging?
6. What is defensive programming?
7. How does binary search work?
8. What is the time complexity of selection sort?
9. What is Big O notation?
10. What is quantum entanglement?

## Scoring

Score each answer from 0 to 2 for each criterion:

- 0 = Poor / incorrect / unsupported
- 1 = Partially correct
- 2 = Fully satisfactory

Maximum score per question:

6 criteria x 2 = 12 points

Maximum total score:

10 questions x 12 = 120 points

## Recorded manual evaluation

A decomposition answer was manually evaluated using the six criteria.

Recorded score:

11/12

This is one manually evaluated answer and should not be interpreted as a statistically significant benchmark for the full question set.

## Evaluation status

The current retrieval evaluation covers all 10 questions and measures whether the expected source is retrieved above the configured similarity threshold.

The answer-quality evaluation is a separate manual assessment and has not yet been completed for all 10 questions.

## Interpretation

Retrieval success and answer quality are different measurements.

A question can retrieve the expected document source while the generated answer may still require evaluation for:

- relevance
- grounding
- correctness
- completeness
- hallucination avoidance
- source-reference quality