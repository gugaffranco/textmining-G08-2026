"""
Reusable helper functions for the project.

Anything used in more than one notebook (EDA, preprocessing, classification,
sentiment analysis) lives here and is imported with:

    from utils.text_utils import <function>

Each function below is a stub: replace `raise NotImplementedError` with the
implementation when working on the matching GitHub issue.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path_train="datasets/train.csv", path_test="datasets/test.csv"):
    """Load train and test sets and return (train_df, test_df)."""
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Row filtering (train only)
# ---------------------------------------------------------------------------

def drop_non_lyrics(df):
    """Drop rows whose lyrics are empty, URL-only or Genius placeholders."""
    raise NotImplementedError


def drop_duplicate_lyrics(df):
    """Drop conflicting-tag duplicate groups, then keep one copy per lyric."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Text cleaning (train and test)
# ---------------------------------------------------------------------------

def remove_urls(text):
    """Remove URLs from a lyric string."""
    raise NotImplementedError


def remove_section_markers(text):
    """Remove Genius markers such as [Verse 1: X], [Chorus], *inaudible*, (x2)."""
    raise NotImplementedError


def normalize_text(text):
    """Lowercase, expand contractions, handle punctuation and whitespace."""
    raise NotImplementedError


def clean_lyrics(text):
    """Full cleaning pipeline for one lyric string (calls the functions above)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def fix_year(df, min_year=1500, max_year=2026):
    """Set impossible years to NaN instead of dropping the rows."""
    raise NotImplementedError


def parse_features(value):
    """Parse the set-like `features` string into a Python list of artists."""
    raise NotImplementedError
