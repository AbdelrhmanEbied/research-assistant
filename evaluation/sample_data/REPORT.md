# RAG Evaluation Report

## Config

- Dataset: `evaluation/sample_data/example.jsonl`
- Store: `(temp corpus)`
- Top-k: `3`
- Search types: `dense, sparse, hybrid`
- Rerank: `False`
- LLM judge: `False`

## Retrieval metrics (average)

| config | hit_rate | recall | precision | mrr | ndcg | n |
|---|---|---|---|---|---|---|
| dense/no-rerank:chunk | 0.950 | 0.950 | 0.367 | 0.900 | 0.909 | 20 |
| dense/no-rerank:document | 0.950 | 0.950 | 0.525 | 0.925 | 0.932 | 20 |
| hybrid/no-rerank:chunk | 1.000 | 0.975 | 0.367 | 0.900 | 0.910 | 20 |
| hybrid/no-rerank:document | 1.000 | 1.000 | 0.525 | 0.950 | 0.959 | 20 |
| sparse/no-rerank:chunk | 1.000 | 0.975 | 0.367 | 0.942 | 0.936 | 20 |
| sparse/no-rerank:document | 1.000 | 1.000 | 0.492 | 1.000 | 0.996 | 20 |

JSON report written to evaluation/sample_data/report.json
