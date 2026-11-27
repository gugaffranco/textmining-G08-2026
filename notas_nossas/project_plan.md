# Project plan

**Deadline:** 22 December 2026, 17:59. Each day early earns +0.15 points (up to +1 point). Each day late costs −1 point. **Target:** 19 December.

Who does what, branches and milestones: see the [README](../README.md). Tasks are GitHub issues.

## 1. Workflow

The workflow is a loop, not a straight line:

```
question → look at the data → clean → EDA ─┐
               ▲                           │ EDA finds new problems
               └───────── clean again ◄────┘
                              │
                              ▼
                   apply techniques → share insights
```

Example: the `[?]`, `[x2]` and `[Guitar Solo]` tags were only found by exploring, and they meant more cleaning.

## 2. Two tasks, two pipelines

| | Genre identification | Sentiment analysis |
|---|---|---|
| Type | Supervised (we have genre labels) | No sentiment labels, so a lexicon or pretrained model |
| Data | Train on `train.csv`, predict `test.csv` (no `tag` column) | `train.csv` only (true genre, year and views are all there) |
| Output | Kaggle submission | Answers to our sentiment questions |
| Needs the other task? | No (idea: sentiment score as an extra feature) | No, it uses the real genres, not predicted ones |

The two tasks share data loading, EDA and section splitting. After that they need **different preprocessing**.

## 3. Rules about order (and why)

| Rule | Why | Example |
|---|---|---|
| Split into sections **before** cleaning | Cleaning removes the brackets, so we would lose where each chorus starts | `[Chorus]` → `Chorus` → just a word |
| POS tagging **before** lemmatization | The lemmatizer needs to know if a word is a noun or a verb | "Maria leaves Lisbon": without POS, "leaves" becomes "leaf" |
| POS tagging **before** stopword removal | The tagger uses words like "the" as context | "the leaves are falling" → "leaves falling": noun or verb? |
| Do not remove negations before sentiment | The meaning flips | "I don't love you no more" → "love" (positive!) |

## 4. Shared steps (both tasks)

- [ ] Load data and look at it (data description for the report)
- [ ] Split lyrics into sections using the tags (see `sa_chorus_vs_verse.md`). This also answers the brief's question: *"Did you use the entire song or split sections into different columns?"*
- [ ] Clean the tags: catch variants (`[Verse One]`, `[Hook x2]`, `[Pre Chorus]`), and put `[?]`, `[Instrumental]` and similar tags in "other"
- [ ] EDA that leads to decisions (the brief says: no charts that don't help answer the problem)

## 5. Genre pipeline

**Decisions so far**
- Slang marks the genre ("ain't", "y'all", "gonna"). **Do not lemmatize** (to be confirmed by an ablation), because it would erase that signal.
- Without lemmatization, **POS tagging is not needed**.
- Representation: TF-IDF. Common words like "the" get a weight close to zero, so removing stopwords matters less than with plain counts.

**Evaluation**
- Score on a part of `train.csv` the model never saw (holdout or cross-validation). Never score on the same songs used to train.
- Metric: **macro F1**, not accuracy. The classes are unbalanced (approximate shares, to confirm in EDA: pop ~41%, rap ~29%, rock ~19%, rb ~5%, misc ~4%, country ~3%). A model that always says "pop" gets ~41% accuracy and finds zero country songs.
- [ ] Check which metric the Kaggle competition uses

**Ablations** (change one thing at a time and compare macro F1)
- [ ] Stopwords removed vs kept
- [ ] Lemmatization vs none
- [ ] Slang protected vs not
- [ ] TF-IDF vs plain counts
- [ ] (Creativity) Add a sentiment score as a feature

## 6. Sentiment pipeline

**Decisions so far**
- Custom stopword list that **keeps negations** ("no", "not", "never", "n't", ...)
- If we lemmatize, the order is: POS → stopword removal → lemmatization

**Questions to answer**
- [ ] (a) Main sentiments per genre
- [ ] (b) Change in sentiment over the years
- [ ] (c) Does sentiment affect popularity (views)?
- [ ] (extra) Chorus vs verse sentiment (see `sa_chorus_vs_verse.md`)

## 7. Open questions

- **Slang and tokenization:** does NLTK's tokenizer or punctuation removal break "y'all", "ain't" and "don't"? How do we protect them: keep apostrophes, use a tokenizer built for informal text, or check the stopword list?
- **Sentiment tool:** a lexicon or a pretrained model? What input does it expect (raw sentences or cleaned tokens)? If it expects raw sentences, should we skip stopword removal for sentiment?
- **misc genre:** what is in it, and how do we handle it?

## 8. Limitations to report

- Genre labels are probably user-tagged, so some are likely wrong. The classifier learns from those same labels and can't fix them.
- Section tags are more common in recent songs (2000s: 18%, 2020s: 44%), so the chorus vs verse analysis is biased toward recent songs.
- Everything in the notebook must be something we can explain. The professors will ask about it.
