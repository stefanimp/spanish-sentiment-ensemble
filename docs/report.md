# Spanish Sentiment Ensemble

### stefanimp · Technical report · 23 October 2026

Four-class Spanish sentiment analysis using complementary transformer models and a
hierarchical cascade for opinion boundaries.

## Outcome and task

| Measure | Result | Source |
| --- | --- | --- |
| Tournament position | First place | Author confirmation |
| Public macro F1 | 0.64182 | Author confirmation and explicit public-score clarification |
| Training set | 1,707 posts | Archived corpus |
| Test set | 1,706 posts | Archived corpus |
| Replay parity | All 1,706 labels match | Comparison with archived v22_02 CSV |

This individual University of Jaén project predicts positive (P), negative (N), neutral (NEU)
or no opinion (NONE) for Spanish social posts. Macro F1 gives each class equal weight. The
training set contains 522 P, 741 N, 223 NEU and 221 NONE examples, so aggregate accuracy alone
would obscure performance on the smaller classes.

The approach combines RoBERTuito, Cardiff, BETO and a cascade that explicitly separates
absence of opinion, neutral opinion and polarity. The aim is useful diversity of errors rather
than increasing the size of a single model.

## Result provenance

The author confirms first place and identifies 0.64182 as a public score. No private score
or PLN leaderboard capture has been supplied. The archived academic report contains other
experimental figures, including 0.66769 for v22_02. Those figures are not substituted for the
author-confirmed result. Available evidence does not conclusively identify the winning CSV.

Reproducing v22_02 verifies the saved ensemble formula and output order. It does not recalculate
macro F1 or establish that this particular archived CSV obtained first place.

---

## Models and training recipe

| Component | Backbone | Purpose |
| --- | --- | --- |
| RoBERTuito | pysentimiento/robertuito-base-uncased | Spanish social language |
| Cardiff | cardiffnlp/twitter-xlm-roberta-base-sentiment | Multilingual social sentiment |
| BETO | dccuchile/bert-base-spanish-wwm-cased | Spanish BERT representation |
| Cascade | pysentimiento/robertuito-sentiment-analysis | Binary opinion boundaries |

The retained base recipes use 128-token inputs, learning rate 2e-5, four epochs, class-weighted
cross-entropy and label smoothing 0.10. RoBERTuito uses lowercase social-text normalization;
Cardiff and BETO preserve capitalization. URLs, mentions, hashtags, emojis and repeated
punctuation are normalized consistently with the archived recipes.

The cascade trains NONE/rest, NEU/polarity and P/N boundaries for 4, 5 and 3 epochs respectively,
with smoothing 0.05. Their probabilities are composed as follows:

```text
p(NONE) = p_none
p(NEU)  = (1 - p_none) * p_neu
p(P)    = (1 - p_none) * (1 - p_neu) * p_positive
p(N)    = (1 - p_none) * (1 - p_neu) * (1 - p_positive)
```

The composed probabilities are clipped at 1e-12, normalized and saved as float32 log scores.
The final ensemble casts inputs to float64 and preserves the original operation order:

```text
L = 0.40 * RoBERTuito + 0.31 * Cardiff + 0.19 * BETO + 0.10 * Cascade
scores = normalize(softmax(L) * [0.97, 1.00, 1.08, 1.08])
prediction = argmax(scores)                      # P, N, NEU, NONE
```

The class-adjusted scores have not been established as calibrated confidence estimates.

---

## Reproduction and maintenance

The CPU replay package needs only NumPy. Standard-library CSV handling preserves string IDs,
row order and archived CRLF output. Matrix validation rejects malformed shapes, missing
components and non-finite values. An optional reference must match before output is written.
Existing outputs are rejected rather than overwritten.

```text
uv sync --python 3.12 --locked --extra dev
uv run spanish-sentiment demo
uv run pytest -q
```

The demo is synthetic. Full replay uses four original logit matrices and test IDs, excluded
from Git. The CLI also provides explicit training recipes and a dry-run mode that validates
input without downloading models or starting training. PyTorch and Transformers are optional
and loaded only by the training backend.

Regression checks cover the archived 1,706-row output and a synthetic original-formula fixture.
All 6,826 preprocessing comparisons across both modes and the original train/test rows also match.
The training backend is checked offline with a tiny randomly initialized BERT model, including
a weighted loss, gradient accumulation and prediction. Full transformer GPU training was not
repeated during this maintenance pass.

## Limits and data audit

The original data has one repeated training ID, one ID shared between train and test, and two
exact text overlaps across those splits. The shared ID also has identical text. Rows are
preserved for historical reproduction; a decontaminated evaluation would be a separate study.

Small data and repeated leaderboard-guided selection limit the strength of generalization
claims. The original experiment did not pin immutable backbone revisions. A future training
run may differ with hardware, model revisions or library changes, even when it uses seed 42.

The author is stefanimp. The maintenance edition was prepared with AI assistance.
Sources are the university InterTASS brief, original code and artifacts, author confirmation
and local regression checks. Corpus reference: Díaz-Galiano et al., Overview of TASS 2019,
IberLEF@SEPLN, pp. 550–560. Code uses MIT; upstream models and datasets retain their own terms.
