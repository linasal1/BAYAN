# Lab Notes

## Lab 1 — Defect Safari
### Sentence segmentation spot-check

- English multi-sentence example: segmented correctly.
- Arabic multi-sentence example: segmented correctly.
- Numbered-list complaint: spaCy split the list numbers (`1.`, `2.`, `3.`) into separate sentence fragments.
- Finding: basic sentencizer works for ordinary Arabic/English sentences, but numbered lists may require an additional rule or preprocessing guardrail.


### Defect 1
-  ⁠Class: Extra whitespace
- ⁠Example: `  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   `
- ⁠Why it matters: Extra whitespace creates inconsistent text representations without adding useful meaning.
- ⁠Decision: Remove unnecessary leading and trailing whitespace.


### Defect 2
- ⁠Class: HTML remnants
- ⁠Example: ⁠ الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة <br> ⁠
-  ⁠Why it matters: HTML tags are formatting noise and do not add useful linguistic meaning.
- ⁠Decision: Remove HTML tags while preserving the surrounding text.


### Defect 3
- ⁠Class: Phone-number PII
- ⁠Example: ⁠ تطبيق خدمات المياه يتوقف عند تسجيل الدخول 0551234567 ⁠
- ⁠Why it matters: Phone numbers are personally identifiable information and should not be exposed to the model.
- ⁠Decision: Replace phone numbers with ⁠ <PHONE> ⁠.


### Defect 4
- Class: National-ID-shaped PII
- ⁠Example: ⁠ إنارة الممر لا تعمل عند طريق الملك فهد 1023456789 ⁠
- Why it matters: National-ID-shaped values may contain sensitive personal information and should not be exposed to the model.
- ⁠Decision: Replace these values with ⁠ <NATIONAL_ID> ⁠.


### Defect 5
- ⁠Class: Emoji
- ⁠Example: ⁠ انقطاع المياه مستمر في حي العليا منذ الصباح 😡 ⁠
- ⁠Why it matters: Emoji can carry useful sentiment information, so removing it may remove task-relevant signal.
- ⁠Decision: Preserve emoji.


### Defect 6
- ⁠Class: Excessive character repetition
- ⁠Example: ⁠ لووووسمحت ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة 😡 ⁠
- ⁠Why it matters: Repeated characters may represent emphasis or sentiment and can also affect tokenisation.
- ⁠Decision: Treat repetition according to the preprocessing contract instead of removing it blindly.


## Lab 2 — Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | 177,853,440 | 51.84% | More than half of the model parameters are in embeddings, reflecting the cost of supporting a large multilingual vocabulary. |
| CAMeLBERT | 109,081,344 | 21.48% | Much smaller embedding share because the vocabulary is Arabic-focused; a larger share of parameters is used in the transformer layers. |

### Observation
mBERT has a much larger embedding parameter share than CAMeLBERT. This suggests a multilingual vocabulary tax: supporting many languages increases the size of the embedding matrix.

### Attention masking

- Causal masking correctly prevented tokens from attending to future positions.
- PAD attention without a mask was 0.172933 (~17.3%).
- PAD attention with the mask was 0.000000.
- Conclusion: without an attention mask, the model can waste attention on padding tokens; masking removes this leakage.

### Real attention inspection

- Arabic and English attention maps were generated successfully.
- mBERT split some Arabic and English words into subword pieces.
- The attention matrices were 10×10 for Arabic and 11×11 for English.
- Attention maps are useful for diagnostics, but they should not be treated as proof of why the model made a decision.


## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
