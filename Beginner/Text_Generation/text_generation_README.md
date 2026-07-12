# Text Generation — Hugging Face Pipeline

Uses `pipeline('text-generation', model="gpt2")` to autocomplete text, tried first on lines from
Robert Frost poems (`robert_frost.txt`), then on a plain prompt sentence.

## Setup
- `gen = pipeline('text-generation', model="gpt2")` — loads GPT-2 (small), a decoder-only model
  that predicts the next token given everything before it.
- `set_seed(1234)` — makes sampling reproducible; without it, repeated calls give different
  completions since generation samples from a probability distribution, not just argmax.
- `lines = [line.strip() for line in open('robert_frost.txt')]`, then drop empty lines — raw
  poem text has blank lines between stanzas that aren't useful as prompts.

## Core call
`gen(prompt, **kwargs)` returns a list of dicts: `[{'generated_text': "..."}]` — one dict per
sequence requested. The `generated_text` includes the prompt itself plus the model's continuation.

## Key parameters
| Param | What it does | Why it matters |
|---|---|---|
| `max_length` | Total token budget for prompt + generated text | Too low cuts the output mid-thought; too high runs slower and can drift off-topic |
| `num_return_sequences` | How many independent completions to sample | Lets you compare variety from the same prompt instead of trusting one sample |
| (default sampling) | GPT-2's pipeline samples token-by-token from the predicted distribution | Same prompt + no seed → different output each run; same prompt + seed → reproducible |

## Helper
- `wrap(x)` — `textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)`, just
  reformats long generated text into readable line-wrapped paragraphs for printing.

## Key takeaway from this run
Short `max_length` values (10–20 tokens) on single poem lines mostly produce grammatical but
generic continuations — GPT-2 has no real knowledge of Frost's style or the poem's context, it's
just predicting plausible next words. Feeding it a longer, denser prompt (the "Neural networks
with attention..." NLP sentence) at `max_length=310` produced more coherent, on-topic text,
showing GPT-2 leans heavily on the prompt's own content/style to steer generation rather than any
built-in factual grounding.