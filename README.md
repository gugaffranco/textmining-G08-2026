# Text Mining G08 – Genre and sentiment in song lyrics

Two tasks: **predict a song's genre** from its lyrics (Kaggle), and **analyse sentiment** across genres. Brief: `Text Mining Project 26_27.pdf`. Full plan: [`notas_nossas/project_plan.md`](notas_nossas/project_plan.md).

**Deadline:** 22 Dec 17:59. **Our target:** submit on 19 Dec (+0.15 points per day early).

## Tracks

| Track | Owner | Branch | Notebook | Label |
|---|---|---|---|---|
| EDA | Lince | `EDA` | `01_Data_Exploration_Preprocessing.ipynb` | `EDA` |
| Shared preprocessing | Simão | `preprocessing` | `02_Preprocessing.ipynb` | `Preprocessing` |
| Genre classification | TBD | `classification` | `03_Genre_Classification.ipynb` | `Classification` |
| Sentiment analysis | TBD | `sentiment` | `04_Sentiment_Analysis.ipynb` | `Sentiment` |
| Report | everyone | – | inside the notebooks | `Report` |

Reusable functions go in `utils/` (`EDA.py`, `preprocessing.py`, `classification.py`, `sentiment.py`).

Classification and sentiment don't have to wait for preprocessing: start on the raw lyrics and plug in the cleaning when it's ready.

## Milestones

1. **EDA + shared preprocessing**, by 6 Dec
2. **Classification + sentiment**, by 13 Dec
3. **Report + final run**, by 18 Dec

## How we work

1. Pick an issue, assign yourself, work on **your track's branch**.
2. **Only the owner edits a notebook.** Notebooks merge badly; `utils/*.py` files merge fine.
3. Clear notebook outputs before committing (except the final run).
4. Open a **small PR into `main`** often, write `Closes #<issue>` in it, and ask a teammate to review.
5. `main` must always run from start to finish.
6. Before starting work: `git pull origin main` into your branch.

## Setup

```bash
conda activate text-mining1
git lfs pull          # downloads the real train.csv / test.csv
```

Each notebook downloads the NLTK data it needs in its first cell (both `punkt` and `punkt_tab`, so it runs on any NLTK version).

## Rules from the brief

- Models: only vanilla scikit-learn, TensorFlow/Keras, PyTorch or Transformers. **No XGBoost, LightGBM, CatBoost or AutoML** (−1 point each time).
- Cite peer-reviewed sources only. No Medium, TowardsDataScience or LLMs.
- Every cell needs a clear purpose. Don't explain theory from class.
- Only include what everyone can explain in the oral discussion.
