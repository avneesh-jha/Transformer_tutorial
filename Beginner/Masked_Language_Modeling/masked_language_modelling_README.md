# Denoising Autoencoder (DAE) for Text as a Masked Language Model (MLM)

A **Denoising Autoencoder (DAE)** used for text as a **Masked Language Model (MLM)** is a neural network architecture designed to reconstruct clean text from intentionally corrupted input by predicting hidden or altered tokens. This is the foundational pre-training concept behind state-of-the-art transformer models like **BERT** and **BART**.

---

## 1. The Core Concept

In image processing, "noise" means adding random pixel variations or setting pixels to 0. In text-based MLM, "noise" means **corrupting tokens (words/subwords)**.

Setting a pixel to 0 maps directly to replacing a word with a special `[MASK]` token, stripping away its visual/semantic information so the network is forced to guess it based strictly on surrounding context.

```text
Original Input Text:     "The cat sat on the mat"
                             ↓
Noising Process (MLM):   "The cat [MASK] on the mat"  (Equivalent to "setting to 0")
                             ↓
Encoder-Decoder Stack:   [ Neural Network Processing ]
                             ↓
Model Output Target:     Predict: "[MASK] = sat"
```

---

## 2. How the "Noise" is Applied

Following the standard BERT recipe, noise is typically applied to roughly **15% of the input tokens**. That chosen 15% is corrupted using three distinct strategies to make the model robust:

* **80% Masking (The "Set to 0" equivalent):** The token is replaced with `[MASK]`. This forces the network to learn contextual syntax and semantics to fill in the blank.
* **10% Random Replacement:** The token is replaced with a completely random word from the vocabulary. This forces the model to remain vigilant because any word might be wrong, preventing it from relying *only* on the presence of a `[MASK]` token.
* **10% Left Unchanged:** The token is kept exactly as it is. This biases the representation toward the actual observed word, ensuring the model preserves correct contextual features.

---

## 3. The Architecture Workflow

1. **Tokenization & Embedding:** Text is split into tokens, converted into vectors, and injected with positional embeddings.
2. **The Encoder (Corrupted Representation):** Self-attention layers process the noisy input sentence. Because it cannot see the masked words, it must use the surrounding words (**bidirectional context**) to infer what is missing.
3. **The Decoder / Prediction Head:** A linear layer with a softmax activation sits on top of the encoder output. It calculates a probability distribution over the entire vocabulary *only* for the tokens that were corrupted.
4. **Loss Calculation:** The model uses **Cross-Entropy Loss** to compare its predictions against the original, uncorrupted words.

---

## 4. Why Use DAE as an MLM?

* **Bidirectional Context:** Unlike auto-regressive models (like GPT) that only look left-to-right, an MLM looks both ways simultaneously to understand deep semantic meaning.
* **Unsupervised Pre-training:** It eliminates the need for human-labeled data. You can scrape massive text corpora, randomly drop noise into sentences, and let the model teach itself grammar, facts, and reasoning.
* **Downstream Transfer:** Once training is complete, the decoder head is discarded, and the encoder is fine-tuned for tasks like sentiment analysis, text classification, or question-answering.

---

## 5. Notebook Walkthrough

### Data setup
- Downloads `bbc_text_cls.csv` (BBC news articles with a `labels` column: business, tech, sport, etc.).
- `label = "business"` → filters `df` down to business articles only.
- `np.random.seed(1234)` + `np.random.choice(texts.shape[0])` — picks one random article
  (`doc`) reproducibly, so the same document is used every run.
- `textwrap.fill(doc, ...)` — just pretty-prints the long raw string into wrapped lines for
  readability, same helper pattern as the text-generation notebook.

### `pipeline("fill-mask")`
- No model name passed → Hugging Face defaults to **DistilRoBERTa-base**, an encoder-only MLM.
  This is the practical, ready-trained stand-in for the DAE/MLM theory above — masking has
  already happened during its pre-training; here we just query it.
- Its mask token is `<mask>` (RoBERTa-style), **not** `[MASK]` (BERT-style) — an important
  detail if you swap models, since the mask literal is model/tokenizer-specific
  (`mlm.tokenizer.mask_token` is the safe way to get it programmatically, as used later in
  the exercise code).
- Output shape: a list of dicts, one per candidate fill, each with `score` (confidence),
  `token`, `token_str` (the predicted word), and `sequence` (the full sentence with that
  word substituted in) — ranked by score, highest first.

### Single-mask experiments
The notebook masks the **same sentence** at four different positions one at a time:
```
"Shares in <mask> and plane-making giant Bombardier have fallen to a 10-year low
 following the departure of its chief executive and two members of the board."
```
| Masked word | What the model has to infer from context |
|---|---|
| `<mask>` and plane-making giant... | An industry/company noun (paired with "plane-making") |
| ...following the `<mask>` of its chief executive | An event noun (e.g. "departure", "resignation") |
| ...of its chief `<mask>` and two members | A job title (e.g. "executive") |
| ...two `<mask>` of the board | A collective noun (e.g. "members") |

This demonstrates the **bidirectional context** point from Section 4 directly: each prediction
is only as good as the surrounding words on *both* sides, and moving the mask to a new position
changes what the model is even trying to solve.

### Exercise — automatic TF-IDF-based document masking
The exercise cell ("write a function that automatically masks and replaces words in a whole
document... based on some statistic, e.g. TF-IDF") is solved in the cells that follow it:

**1. Build a TF-IDF vocabulary → IDF lookup**
```python
tfidf = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
tfidf.fit(df['text'])
idf_map = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
```
- Fit over the **whole corpus** (`df['text']`, all labels), not just the one document, so IDF
  reflects how rare/informative a word is across all BBC articles, not just this one.
- `stop_words='english'` strips common words (the, and, of...) so they're never mask candidates.
- Only the **IDF** component is used (not full TF-IDF) — `idf_map` is a static word → rarity
  score dictionary. Rarer, more topic-specific words (e.g. company names, technical terms) get
  higher scores than generic words.

**2. `mask_and_replace_sentence(sentence, mlm, idf_map, pct=0.15, min_word_len=3)`**
- Extracts candidate words with a regex (`word_re`), filters to unique words of at least
  `min_word_len` characters.
- Scores each candidate by its corpus-level IDF (`idf_map.get(w.lower(), 0)` — unseen words
  default to 0, i.e. never prioritized).
- Picks the **top `pct` fraction** (default 15%, matching the BERT recipe in Section 2) of
  candidates ranked by IDF score — i.e. it masks the *most distinctive/rare* words in the
  sentence, not a random 15% like real BERT pre-training.
- For each chosen word: replaces its first occurrence with `mlm.tokenizer.mask_token`, runs
  the `fill-mask` pipeline, and substitutes in the **top-ranked prediction**
  (`preds[0]['token_str']`) — so the sentence is rewritten word-by-word, each replacement
  informed by the (partially already-edited) sentence context.
- Returns the sentence unchanged if there are no valid candidates.

**3. `mask_and_replace_document(doc, mlm, idf_map, pct=0.15, min_word_len=3)`**
- Splits the document into paragraphs (`\n\n`), then each paragraph into sentences via a
  punctuation-lookahead regex (`(?<=[.!?])\s+`).
- Runs `mask_and_replace_sentence` independently per sentence, then rejoins sentences with
  spaces and paragraphs with `\n\n` — preserves the original document structure.

**4. Final demo** — runs the whole pipeline on the randomly chosen business article and prints
`ORIGINAL` vs `MODIFIED` side by side via `textwrap.fill`.

## Key takeaway from this run
Targeting high-IDF (rare/topic-specific) words for masking — instead of BERT's random 15% —
means the rewritten document keeps its grammar and generic wording intact, while the words that
carry the most topic-specific meaning (names, specific nouns) get swapped for the model's best
contextual guess. This is a good proxy for testing how much the MLM actually "understands" a
domain versus how well it just handles boilerplate sentence structure: a fluent but factually
drifted `MODIFIED` output shows the model can preserve grammaticality (its bidirectional
training objective) even when it gets the specific facts wrong (it was never trained to be
factually grounded, just contextually plausible).