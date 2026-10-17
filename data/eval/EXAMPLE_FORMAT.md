# Eval set format (example only)

`data/eval/qa_set.jsonl` holds the hand-written evaluation questions, **one JSON object per line**.
This file only shows the shape. Every value below is a placeholder, not a real question. The
validator never reads this file (it reads only `qa_set.jsonl`).

Validate with `python -m fincrime_rag.evalset validate`. Look up page numbers with
`python -m fincrime_rag.evalset pages <doc_id>`: gold pages are **PDF page indexes** (the `pdf_page`
column), not printed page numbers.

## Fields

| field | required | values |
|---|---|---|
| `qid` | yes | unique string, e.g. `"q01"` |
| `question` | yes | the question text (no duplicates) |
| `type` | yes | `answerable`, `unanswerable`, `injection` |
| `expected_behavior` | yes | `answer` (for answerable), `refuse` (unanswerable), `resist_injection` (injection) |
| `category` | yes | `lookup`, `definition`, `list`, `numeric`, `multi_page`, `multi_doc`, `conflict`, `paraphrase`, `false_premise`, `out_of_corpus`, `injection` |
| `gold` | yes | list of `{"doc_id": <manifest doc_id>, "pages": [<pdf_page>, ...]}`; non-empty for answerable, empty list for the other two types |
| `reference_answer` | yes | short reference answer (for a refusal or injection item: what a correct response looks like) |
| `key_facts` | yes | list of 1-4 short strings the answer must contain |
| `split` | yes | `dev` or `test` |
| `notes` | no | free text |

## Shape of a record (placeholders)

Written across several lines here for readability; in the real file each record is **one line**.

```json
{
  "qid": "<unique id>",
  "question": "<question text>",
  "type": "answerable",
  "expected_behavior": "answer",
  "category": "<one of the categories above>",
  "gold": [{"doc_id": "<doc_id from data/manifest.csv>", "pages": [0]}],
  "reference_answer": "<reference answer>",
  "key_facts": ["<fact>", "<fact>"],
  "split": "dev",
  "notes": "<optional>"
}
```

For `type` = `unanswerable` or `injection`, `gold` is `[]` and `expected_behavior` is `refuse` or
`resist_injection`. The `0` above is a deliberately invalid page placeholder.

## Rules the validator enforces

- Errors: unique `qid`; no duplicate questions; valid `type`, `category`, `expected_behavior`,
  `split`; `type` and `expected_behavior` consistent; every `doc_id` in `data/manifest.csv`; every
  page within the document's page count; `gold` non-empty for answerable and empty otherwise;
  `key_facts` count within the limits in `config.yaml` (`evalset:`); no unknown fields.
- Warnings (do not fail): a gold page tagged non-body (for example Reg E model forms); more than
  3 questions on one document; a document with no questions; `multi_doc` with fewer than two gold
  documents; `multi_page` with no document that has two or more gold pages.
