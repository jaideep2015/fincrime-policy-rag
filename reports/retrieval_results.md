# Retrieval results (LLM-free)

Splits scored: dev; answerable questions only; **n = 14**. Chunking 512/64 tokens, embedding `BAAI/bge-small-en-v1.5`, RRF k = 60, candidate pool 50.

`any@k`: a top-k hit covers at least one gold page. `frac@k`: share of gold pages covered in the top k. `MRR`: reciprocal rank of the first covering hit. `doc-hit@5`: a gold document appears in the top 5 (page ignored). With this n, per-category rows rest on very few questions and differences between modes are not statistically established.

| Mode | Slice | n | any@1 | any@3 | any@5 | any@10 | frac@1 | frac@3 | frac@5 | frac@10 | MRR | doc-hit@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | overall | 14 | 0.64 | 0.86 | 0.93 | 1.00 | 0.38 | 0.63 | 0.74 | 0.85 | 0.756 | 1.00 |
| dense | conflict | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.29 | 0.46 | 0.46 | 0.71 | 1.000 | 1.00 |
| dense | false_premise | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.00 |
| dense | list | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.167 | 1.00 |
| dense | lookup | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.67 | 1.00 | 1.00 | 1.000 | 1.00 |
| dense | multi_doc | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 | 0.80 | 1.00 | 1.00 | 1.000 | 1.00 |
| dense | multi_page | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.75 | 0.75 | 0.75 | 0.667 | 1.00 |
| dense | numeric | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.75 | 0.75 | 0.75 | 0.667 | 1.00 |
| dense | paraphrase | 4 | 0.50 | 0.75 | 1.00 | 1.00 | 0.50 | 0.62 | 0.88 | 0.88 | 0.688 | 1.00 |
| bm25 | overall | 14 | 0.57 | 0.79 | 0.86 | 0.86 | 0.27 | 0.62 | 0.74 | 0.80 | 0.693 | 0.86 |
| bm25 | conflict | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.29 | 0.58 | 0.75 | 0.88 | 1.000 | 1.00 |
| bm25 | false_premise | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.00 |
| bm25 | list | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 |
| bm25 | lookup | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.67 | 1.00 | 1.00 | 1.000 | 1.00 |
| bm25 | multi_doc | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 | 0.80 | 0.80 | 1.00 | 1.000 | 1.00 |
| bm25 | multi_page | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.25 | 0.25 | 0.75 | 1.00 | 0.600 | 1.00 |
| bm25 | numeric | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | 1.00 | 1.00 | 0.750 | 1.00 |
| bm25 | paraphrase | 4 | 0.25 | 0.75 | 0.75 | 0.75 | 0.12 | 0.62 | 0.62 | 0.62 | 0.500 | 0.75 |
| hybrid | overall | 14 | 0.71 | 0.86 | 0.86 | 1.00 | 0.40 | 0.63 | 0.71 | 0.96 | 0.796 | 0.93 |
| hybrid | conflict | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.29 | 0.46 | 0.58 | 1.00 | 1.000 | 1.00 |
| hybrid | false_premise | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.00 |
| hybrid | list | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.167 | 1.00 |
| hybrid | lookup | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.67 | 1.00 | 1.00 | 1.000 | 1.00 |
| hybrid | multi_doc | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | 0.80 | 0.80 | 1.00 | 1.000 | 1.00 |
| hybrid | multi_page | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.75 | 0.75 | 1.00 | 0.750 | 1.00 |
| hybrid | numeric | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.75 | 1.00 | 1.00 | 0.667 | 1.00 |
| hybrid | paraphrase | 4 | 0.75 | 0.75 | 0.75 | 1.00 | 0.62 | 0.62 | 0.62 | 0.88 | 0.786 | 0.75 |
