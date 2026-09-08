# Decision Records

## tokenizer
## Lab 1 — Tokenizer decision

Chosen tokenizer: XLM-R

Reason:
XLM-R provided the best overall bilingual balance on Bayan data. It achieved Arabic fertility of 1.67 and English fertility of 1.43, with p95 sequence lengths of 21 tokens for Arabic and 23 tokens for English.

CAMeLBERT performed better for Arabic alone, with fertility 1.40 and Arabic p95 length 20, but its English fertility was much worse at 2.70.

DistilBERT was efficient for English but fragmented Arabic heavily, with Arabic fertility 4.52 and p95 length 47.

Therefore, XLM-R is the most suitable single tokenizer for Bayan's bilingual Arabic/English pipeline.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
