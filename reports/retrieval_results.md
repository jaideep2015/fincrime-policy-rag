# Retrieval results (LLM-free)

Splits scored: dev; answerable questions only; **n = 14**. Chunking 512/64 tokens, embedding `BAAI/bge-small-en-v1.5`, RRF constant k = 60 (score = 1/(k + rank)), candidate pool 50.

> **Note.** Default mode = hybrid was selected on dev aggregates (n=14); differences between modes are 1-2 questions; there is no held-out split. Dense beats hybrid on any@5 (0.93 vs 0.86).

## Page identity

Chunks store the **PDF page index** (field `pdf_pages`, a list; Chroma: `pdf_pages` as "4,5" plus `pdf_page_start` / `pdf_page_end`), taken from each page record's `pdf_page`. The printed page is a separate field (`printed_page_guess`) and is not used for matching. Gold pages in `qa_set.jsonl` are PDF indexes (`gold[].pages`); a hit covers a gold page when `doc_id` matches and the chunk's `pdf_pages` contains it.

## Overall and by category

`any@k`: a top-k hit covers at least one gold page. `frac@k`: share of gold pages covered in the top k. `MRR`: reciprocal rank of the first covering hit. `doc-level hit rate@5`: the right document (any page) is in the top 5; this is not the page-level any@5. Per-category rows rest on very few questions.

| Mode | Slice | n | any@1 | any@3 | any@5 | any@10 | frac@1 | frac@3 | frac@5 | frac@10 | MRR | doc-level hit rate@5 |
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

## Questions with 2+ gold pages: any@k vs all@k

`all@k`: the top k hits cover every gold page.

| Mode | n | any@1 | any@3 | any@5 | any@10 | all@1 | all@3 | all@5 | all@10 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 11 | 0.64 | 0.91 | 0.91 | 1.00 | 0.09 | 0.27 | 0.45 | 0.55 |
| bm25 | 11 | 0.64 | 0.82 | 0.91 | 0.91 | 0.00 | 0.27 | 0.55 | 0.73 |
| hybrid | 11 | 0.73 | 0.91 | 0.91 | 1.00 | 0.09 | 0.27 | 0.45 | 0.91 |

## Chunk share per document

| doc_id | chunks | share of index |
|---|---|---|
| cfr_12_1005_reg_e | 407 | 58.5% |
| fatf_2023_cyber_fraud | 72 | 10.3% |
| fincen_fta_identity | 27 | 3.9% |
| fincen_2025_a003_cmln | 25 | 3.6% |
| fatf_2020_va_red_flags | 24 | 3.4% |
| ffiec_sar_overview | 24 | 3.4% |
| fincen_2022_a002_elder | 22 | 3.2% |
| fincen_fta_elder_efe | 18 | 2.6% |
| fincen_2024_interagency_efe | 17 | 2.4% |
| fincen_2023_alert005_pig_butchering | 14 | 2.0% |
| fincen_2020_a003_covid_mule | 12 | 1.7% |
| ffiec_appendix_f_red_flags | 10 | 1.4% |
| fincen_2021_sar_faqs | 10 | 1.4% |
| cfr_31_1020_320 | 7 | 1.0% |
| fincen_2025_sar_faqs | 7 | 1.0% |

Reg E dominates the index. Dense and BM25 rank chunks across the whole corpus, so a large document can crowd smaller ones out of the top k (for example the 4-page 2025 SAR FAQs, 7 chunks). This is a possible effect, not something measured here; the table above does not isolate it.

## Questions with no gold page in the top 5 (by question ID only)

- dense: q12
- bm25: q01, q12
- hybrid: q01, q12

Reported for information; nothing was changed in response.
