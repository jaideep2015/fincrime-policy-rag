# Retrieval experiments (retrieval only, LLM-free, exact dense search)

**All 20 questions are dev; there is no held-out split. n = 14 answerable questions.** Metrics: `any@k` (some gold page in the top k), `all@k` (every gold page in the top k, union over the top-k chunks), `MRR` and doc-level hit@5 (a gold document in the top 5, page ignored). Page matching uses the stored `pdf_pages`. Intervals are percentile bootstrap 95% CIs over questions (2000 resamples, seed 42). Every configuration uses **exact** dense search (D-032), so the numbers are reproducible; BM25 and RRF (k = 60, pool 50) are unchanged.

## Decision rule (written before running, D-032)

The default stays **512/64 hybrid**. No other configuration is adopted, because n = 14 cannot separate configurations; a configuration that looks better is reported as a hypothesis for held-out testing, not adopted. Chunk sizes above 512 are excluded because bge-small-en-v1.5 truncates input at 512 tokens.

## Index size per configuration

| Configuration | chunks | Reg E share |
|---|---|---|
| 256/32 | 1362 | 60.1% |
| 384/48 | 911 | 59.6% |
| 512/64 | 696 | 58.5% |

## Point estimates (any@k / all@k, MRR, doc-level hit)

| Config | Mode | any@1 | any@3 | any@5 | any@8 | any@10 | all@1 | all@3 | all@5 | all@8 | all@10 | MRR | hit@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256/32 | dense | 0.79 | 0.86 | 0.86 | 0.93 | 0.93 | 0.21 | 0.29 | 0.50 | 0.71 | 0.79 | 0.820 | 0.93 |
| 256/32 | bm25 | 0.57 | 0.71 | 0.79 | 0.86 | 0.86 | 0.00 | 0.29 | 0.50 | 0.64 | 0.71 | 0.661 | 0.93 |
| 256/32 | hybrid | 0.71 | 0.79 | 0.93 | 1.00 | 1.00 | 0.07 | 0.43 | 0.57 | 0.79 | 0.86 | 0.783 | 1.00 |
| 384/48 | dense | 0.57 | 0.86 | 0.93 | 0.93 | 0.93 | 0.14 | 0.36 | 0.64 | 0.79 | 0.86 | 0.720 | 1.00 |
| 384/48 | bm25 | 0.64 | 0.79 | 0.86 | 0.86 | 0.86 | 0.14 | 0.29 | 0.57 | 0.71 | 0.71 | 0.708 | 0.86 |
| 384/48 | hybrid | 0.64 | 0.86 | 0.93 | 0.93 | 0.93 | 0.07 | 0.50 | 0.71 | 0.79 | 0.79 | 0.756 | 1.00 |
| 512/64 | dense | 0.64 | 0.86 | 0.93 | 1.00 | 1.00 | 0.21 | 0.36 | 0.57 | 0.64 | 0.64 | 0.756 | 1.00 |
| 512/64 | bm25 | 0.57 | 0.79 | 0.86 | 0.86 | 0.86 | 0.07 | 0.36 | 0.57 | 0.71 | 0.71 | 0.693 | 0.86 |
| 512/64 | hybrid | 0.71 | 0.86 | 0.86 | 1.00 | 1.00 | 0.21 | 0.36 | 0.50 | 0.86 | 0.93 | 0.796 | 0.93 |

## With 95% CIs (k = 5 and k = 8)

| Config | Mode | any@5 | all@5 | any@8 | all@8 | MRR | hit@5 |
|---|---|---|---|---|---|---|---|
| 256/32 | dense | 0.86 [0.71, 1.00] | 0.50 [0.21, 0.79] | 0.93 [0.79, 1.00] | 0.71 [0.50, 0.93] | 0.820 [0.629, 1.000] | 0.93 [0.79, 1.00] |
| 256/32 | bm25 | 0.79 [0.57, 1.00] | 0.50 [0.29, 0.79] | 0.86 [0.71, 1.00] | 0.64 [0.43, 0.86] | 0.661 [0.452, 0.869] | 0.93 [0.79, 1.00] |
| 256/32 | hybrid | 0.93 [0.79, 1.00] | 0.57 [0.29, 0.86] | 1.00 [1.00, 1.00] | 0.79 [0.57, 1.00] | 0.783 [0.604, 0.946] | 1.00 [1.00, 1.00] |
| 384/48 | dense | 0.93 [0.79, 1.00] | 0.64 [0.43, 0.86] | 0.93 [0.79, 1.00] | 0.79 [0.57, 1.00] | 0.720 [0.530, 0.893] | 1.00 [1.00, 1.00] |
| 384/48 | bm25 | 0.86 [0.71, 1.00] | 0.57 [0.29, 0.79] | 0.86 [0.71, 1.00] | 0.71 [0.50, 0.93] | 0.708 [0.506, 0.899] | 0.86 [0.71, 1.00] |
| 384/48 | hybrid | 0.93 [0.79, 1.00] | 0.71 [0.50, 0.93] | 0.93 [0.79, 1.00] | 0.79 [0.57, 1.00] | 0.756 [0.571, 0.929] | 1.00 [1.00, 1.00] |
| 512/64 | dense | 0.93 [0.79, 1.00] | 0.57 [0.29, 0.86] | 1.00 [1.00, 1.00] | 0.64 [0.36, 0.86] | 0.756 [0.583, 0.917] | 1.00 [1.00, 1.00] |
| 512/64 | bm25 | 0.86 [0.71, 1.00] | 0.57 [0.29, 0.86] | 0.86 [0.71, 1.00] | 0.71 [0.50, 0.93] | 0.693 [0.493, 0.893] | 0.86 [0.71, 1.00] |
| 512/64 | hybrid | 0.86 [0.71, 1.00] | 0.50 [0.29, 0.79] | 1.00 [1.00, 1.00] | 0.86 [0.64, 1.00] | 0.796 [0.626, 0.952] | 0.93 [0.79, 1.00] |

## Paired per-question differences against the baseline (c512_o64 hybrid, exact dense)

Mean of (configuration minus baseline) over the same questions, with a bootstrap 95% CI. An interval that contains 0 means the data cannot tell the configuration from the baseline.

| Config | Mode | any@5 | all@5 | any@8 | all@8 | mrr |
|---|---|---|---|---|---|---|
| 256/32 | dense | +0.00 [+0.00, +0.00] | +0.00 [-0.36, +0.36] | -0.07 [-0.21, +0.00] | -0.14 [-0.43, +0.14] | +0.02 [-0.04, +0.11] |
| 256/32 | bm25 | -0.07 [-0.21, +0.00] | +0.00 [-0.36, +0.36] | -0.14 [-0.29, +0.00] | -0.21 [-0.43, +0.00] | -0.14 [-0.27, -0.03] |
| 256/32 | hybrid | +0.07 [+0.00, +0.21] | +0.07 [-0.14, +0.29] | +0.00 [+0.00, +0.00] | -0.07 [-0.29, +0.14] | -0.01 [-0.15, +0.11] |
| 384/48 | dense | +0.07 [+0.00, +0.21] | +0.14 [-0.21, +0.50] | -0.07 [-0.21, +0.00] | -0.07 [-0.29, +0.14] | -0.08 [-0.24, +0.08] |
| 384/48 | bm25 | +0.00 [+0.00, +0.00] | +0.07 [-0.14, +0.29] | -0.14 [-0.29, +0.00] | -0.14 [-0.29, +0.00] | -0.09 [-0.20, -0.01] |
| 384/48 | hybrid | +0.07 [+0.00, +0.21] | +0.21 [+0.00, +0.43] | -0.07 [-0.21, +0.00] | -0.07 [-0.29, +0.14] | -0.04 [-0.16, +0.09] |
| 512/64 | dense | +0.07 [+0.00, +0.21] | +0.07 [-0.14, +0.29] | +0.00 [+0.00, +0.00] | -0.21 [-0.43, +0.00] | -0.04 [-0.12, +0.02] |
| 512/64 | bm25 | +0.00 [+0.00, +0.00] | +0.07 [-0.14, +0.29] | -0.14 [-0.29, +0.00] | -0.14 [-0.29, +0.00] | -0.10 [-0.21, -0.01] |

## Exact-search baseline next to the published pipeline numbers

| Mode | metric | published (Chroma/HNSW dense, one process) | exact search (this report) | difference |
|---|---|---|---|---|
| dense | any@1 | 0.640 | 0.643 | +0.003 |
| dense | any@3 | 0.860 | 0.857 | -0.003 |
| dense | any@5 | 0.930 | 0.929 | -0.001 |
| dense | any@10 | 1.000 | 1.000 | +0.000 |
| dense | MRR | 0.756 | 0.756 | -0.000 |
| bm25 | any@1 | 0.570 | 0.571 | +0.001 |
| bm25 | any@3 | 0.790 | 0.786 | -0.004 |
| bm25 | any@5 | 0.860 | 0.857 | -0.003 |
| bm25 | any@10 | 0.860 | 0.857 | -0.003 |
| bm25 | MRR | 0.693 | 0.693 | -0.000 |
| hybrid | any@1 | 0.710 | 0.714 | +0.004 |
| hybrid | any@3 | 0.860 | 0.857 | -0.003 |
| hybrid | any@5 | 0.860 | 0.857 | -0.003 |
| hybrid | any@10 | 1.000 | 1.000 | +0.000 |
| hybrid | MRR | 0.796 | 0.796 | -0.000 |

The published numbers come from `retrieval_results.md` and are rounded there (two decimals for any@k, three for MRR), so differences of up to about 0.005 are rounding. BM25 does not use the dense index and differs by the same amount, which shows the size of the rounding. The dense and hybrid rows therefore show no measurable effect of the approximate dense search on these 14 questions, in line with the read-only comparison in D-032 (no differences for the answerable questions in any process).

## Conclusion

The default stays 512/64 hybrid. No configuration is adopted on this evidence. No configuration is better than the baseline with a paired interval that excludes 0. Worse than the baseline with a paired interval that excludes 0: 256/32 bm25 (mrr); 384/48 bm25 (mrr); 512/64 bm25 (mrr). With 40 paired comparisons on 14 questions, a few intervals excluding 0 are expected by chance, so none of this separates configurations; the data cannot tell most of them apart.

## Limitations

- n = 14 answerable questions, all dev, no held-out split: the intervals are wide and a difference of one or two questions is within noise.
- The grid was chosen before running; nothing was tuned on individual questions, and the default pipeline (512/64, hybrid, k = 8, Chroma) was not changed.
- Retrieval only: no claim is made about answer quality.
- Chunk counts differ by configuration, so `any@k` at the same k covers different amounts of text.
