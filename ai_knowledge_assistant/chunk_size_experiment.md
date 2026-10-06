# Chunk Size Retrieval Experiment



## Purpose



Week 3 requires testing different chunk sizes and observing how chunking affects retrieval results.



The experiment used the same:



- embedding model: `all-MiniLM-L6-v2`

- Top-K: `5`

- similarity threshold: `0.35`

- 10-question evaluation set

- expected-source evaluation logic



Only chunk size and overlap were changed.



## Configurations tested



| Chunk size | Overlap | Chunks created | Retrieval pass rate |

|---:|---:|---:|---:|

| 500 | 100 | 152 | 100.0% |

| 1000 | 200 | 77 | 100.0% |

| 1500 | 300 | 51 | 100.0% |



## Observations



### 500 / 100



This configuration produced the largest number of chunks: 152.



All 10 evaluation questions passed retrieval.



The smaller chunks produced relatively high top-1 similarity scores for several questions, including recursion and Big O notation.



### 1000 / 200



This configuration produced 77 chunks.



All 10 evaluation questions passed retrieval.



This is the current production configuration because it has already been integrated and tested end-to-end in both the command-line RAG pipeline and Streamlit application.



### 1500 / 300



This configuration produced 51 chunks.



All 10 evaluation questions passed retrieval.



For the decomposition and abstraction questions, the top-ranked source was the expected decomposition document, unlike the smaller configurations where another semantically related lecture was ranked first.



## Interpretation



All three configurations achieved the same 10/10 retrieval pass rate on the current evaluation set.



Therefore, the experiment does not demonstrate that one configuration is universally more accurate.



The results demonstrate an important RAG trade-off:



- smaller chunks create more retrieval units and can provide more focused passages;

- larger chunks create fewer, broader retrieval units and may preserve more surrounding context;

- chunk size can also affect which document or passage receives the highest similarity score.



## Production decision



The project keeps:



`chunk size = 1000`



`chunk overlap = 200`



This configuration provides a practical balance between chunk count, contextual coverage, retrieval granularity, and the already-tested application behavior.



The choice should be revisited with a larger evaluation dataset before claiming that 1000/200 is globally optimal.



## Important evaluation limitation



The 100.0% pass rate is a retrieval-source metric on the project's 10 predefined questions.



It does not mean:



- all generated answers are 100% correct;

- retrieval will achieve 100% accuracy on unseen questions;

- 1000/200 is mathematically optimal;

- the result generalizes to other document collections.


