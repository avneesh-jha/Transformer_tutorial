# Sentiment Analysis — Hugging Face Pipeline

Uses `pipeline('sentiment-analysis')` (DistilBERT fine-tuned on SST-2) to classify
airline tweets as `POSITIVE`/`NEGATIVE`, then checks how well it matches human labels.

## Data prep
- Load `AirlineTweets.csv`, keep only `text` + `airline_sentiment`.
- Drop `neutral` rows — the model only knows POSITIVE/NEGATIVE, so neutral can't be scored.
- Map labels to numbers: `target = 1` (positive), `0` (negative). Metrics need numeric labels.
- Class counts: 9178 negative vs 2363 positive → **imbalanced**, so accuracy alone is misleading.

## Core values
- `predictions` — raw pipeline output per tweet: `{'label': 'POSITIVE'|'NEGATIVE', 'score': confidence}`.
  `score` is confidence in whichever label won, **not always P(positive)**.
- `probs = score if label is POSITIVE else 1 - score` — normalizes every tweet onto one
  scale (P(positive)), so it's comparable to `target`.
- `preds = 1 if label is POSITIVE else 0` — hard 0/1 class decision (equivalent to `probs > 0.5`).

## Confusion matrix
```
                Predicted Neg   Predicted Pos
Actual Neg           TN              FP
Actual Pos           FN              TP
```
- `confusion_matrix(target, preds)` → **rows = actual class**. Row-normalizing (`normalize='true'`)
  turns rows into **recall per class** (diagonal = TN-rate, TP-rate).
- Swapping args — `confusion_matrix(preds, target)` — swaps rows/cols, turning normalized rows
  into **precision per class** instead. Don't mix the two up.

## Metrics — formula, what it needs, why
| Metric | Formula | Uses | Why |
|---|---|---|---|
| Accuracy | (TP+TN) / all | `preds` | Overall correctness, but hides per-class weakness on imbalanced data |
| Precision | TP / (TP+FP) | `preds` | "When model says positive, how often right" — needs raw counts, not row-normalized `cm` |
| Recall | TP / (TP+FN) | `preds` | "Of real positives, how many caught" — same as diagonal of row-normalized `cm` |
| F1 | harmonic mean of precision & recall | `preds` | Balances both; computed per-class since imbalance makes one class easier |
| ROC-AUC | P(random positive scored higher than random negative) | `probs` | Threshold-independent ranking quality; symmetric under label-flip (good sanity check) |

## Key takeaway from this run
Accuracy (0.890) looked fine but masked a gap: recall/F1 for **positive** tweets (0.846 / 0.759)
was noticeably weaker than for **negative** tweets (0.901 / 0.929) — expected, since the model
was fine-tuned on movie reviews, not tweets, and negative is the majority class. AUC (0.949)
stayed high regardless, showing the underlying ranking is strong even though the fixed 0.5
threshold isn't optimal for the minority (positive) class.
