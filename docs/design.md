# Design notes

## Task and data

The tournament uses Spanish InterTASS posts in four classes. The archived train set contains
1,707 rows: 522 P, 741 N, 223 NEU and 221 NONE. Test contains 1,706 rows. Macro F1 weights
all four classes equally, so performance on the two smaller classes matters.

The data includes one repeated training ID, one ID shared between train and test, and two
exact text overlaps across the splits. The shared ID also has the same text. Training keeps
the original rows to preserve the historical experiment. A decontaminated evaluation would
be a separate experiment, not a silent modification of these results.

## Components and training

| Component | Base model | Role |
| --- | --- | --- |
| RoBERTuito | pysentimiento/robertuito-base-uncased | Spanish social text |
| Cardiff | cardiffnlp/twitter-xlm-roberta-base-sentiment | Multilingual social sentiment |
| BETO | dccuchile/bert-base-spanish-wwm-cased | Spanish BERT representation |
| Cascade | pysentimiento/robertuito-sentiment-analysis | Binary opinion boundaries |

Frozen recipes in `config.py` preserve the selected backbones, 128-token inputs, learning
rate 2e-5, class-weighted cross-entropy and seed 42. Base models use four epochs and label
smoothing 0.10. The cascade uses 4/5/3 epochs for NONE/rest, NEU/polarity and P/N, with smoothing
0.05. Batch size and accumulation reflect the original 4 GB GPU constraints.

Preprocessing replaces URLs and mentions, handles hashtags, spaces emojis and caps repeated
punctuation. RoBERTuito receives lowercase text; Cardiff and BETO retain capitalization.
Training records are kept in their original order, including repeated IDs. Test IDs must be unique.

## Cascade and final decision

```text
p(NONE) = p_none
p(NEU)  = (1 - p_none) * p_neu
p(P)    = (1 - p_none) * (1 - p_neu) * p_positive
p(N)    = (1 - p_none) * (1 - p_neu) * (1 - p_positive)
```

The cascade stores log probabilities as float32 after clipping at 1e-12 and normalizing.
The ensemble casts its four inputs to float64, normalizes weights and sums in fixed order.
Softmax subtracts the row maximum before exponentiation. A class multiplier of
0.97/1.00/1.08/1.08 is applied in P/N/NEU/NONE order before normalization and argmax.
These adjusted scores have not been established as calibrated confidence estimates.

## Result provenance

stefanimp confirms **first place and public macro F1 0.64182**. His explicit clarification
identifies the score as public. No private score or PLN leaderboard capture has been supplied.
The university brief states that final ranking uses the private subset.

The archived report attributes 0.66769 to `v22_02` and lists several similar experimental scores.
That record differs from the author's confirmation. The current presentation uses 0.64182;
the historical figures are retained only in the original local materials.
Available evidence does not tie the confirmed score or first place to a specific archived CSV.

## Verification boundaries

Synthetic fixtures preserve the original ensemble decisions. Full replay matches the 1,706
labels and CSV bytes of `v22_02`; this proves software parity, not a new macro F1 measurement.
Both normalization modes also match the original preprocessing on all 3,413 train/test rows
(6,826 comparisons, with no differences).
The optional training backend is checked offline with a tiny randomly initialized BERT model.
The four full transformers have not been retrained during this maintenance pass. Model revisions,
hardware and dependency changes can affect a future training run.

## Project scope and sources

This is stefanimp's individual university project. The current maintenance edition was
prepared with AI assistance. Original code, logs and reports remain unchanged outside the
current presentation; historical source is retained in a separate local archive.

The [competition](https://www.kaggle.com/competitions/pln-uja-practica-final/overview) and its
university brief identify the task as InterTASS. Corpus reference: Díaz-Galiano et al.,
*Overview of TASS 2019*, IberLEF@SEPLN, pp. 550–560,
[proceedings](https://ceur-ws.org/Vol-2421/). Pretrained models retain their upstream terms.
