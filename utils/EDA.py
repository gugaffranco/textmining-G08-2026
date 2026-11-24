"""
The notebook only calls these functions, so it stays short and readable.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# Settings used in every plot

# Genres ordered from most to least frequent, each one with a fixed colour
Genre_Order = ["pop", "rap", "rock", "rb", "misc", "country"]
Genre_Colors = {
    "pop":"#2a78d6",
    "rap": "#eb6834",
    "rock": "#1baf7a",
    "rb": "#eda100",
    "misc": "#e87ba4", 
    "country": "#008300",
}
Genre_Palete = [Genre_Colors[g] for g in Genre_Order]
MainColor = "#2a78d6"
SecondColor = "#eb6834"

# Year intervals. All the years are kept as they are, even the very old ones
# (the test set can also have texts from those times)
Year_bins = [-np.inf, 1500, 1900, 2000, np.inf]
Year_labels = ["1-1499", "1500-1899", "1900-1999", "2000-today"]

# Regular expressions
URL = r"https?://\S+|www\.\S+"
Section = r"\[\s*(?:verse|chorus|hook|intro|outro|bridge|pre-chorus|refrain|interlude)"
Performer = r"\[[^\]]*:[^\]]+\]"       # [Verse 2: Lil Wayne]
Word = r"(?u)\b[a-z][a-z']+\b"

# Contractions are not in sklearn's stopword list, so we add them
Contractions = {"i'm", "you're", "it's", "don't", "i'll", "we're", "i've", "there's", "you've",
                "that's", "can't", "won't", "didn't", "i'd", "you'll", "we'll", "they're", "let's",
                "isn't", "doesn't", "wasn't", "what's", "he's", "she's", "we've", "you'd",
                "couldn't", "wouldn't"}
EDA_stopwords = list(ENGLISH_STOP_WORDS | Contractions)


def set_plot_style():
    """Same look for all the plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.titleweight"] = "bold" 
    plt.rcParams["axes.spines.top"] = False #just to eleminate the line that is delimitating the graph from the top
    plt.rcParams["axes.spines.right"] = False


# Extra columns used in the EDA
# (no cleaning here: the EDA looks at the raw data, the cleaning is done in preprocessing)

def add_eda_columns(df):
    """Returns a copy of df with descriptive columns: number of words, log of views
    and the year interval (decade) of each song."""
    eda = df.copy()
    eda["word_count"] = eda["lyrics"].str.split().str.len()
    eda["log_views"] = np.log1p(eda["views"])
    eda["year_interval"] = pd.cut(eda["year"], bins=Year_bins, labels=Year_labels, right=False)
    return eda


# 5.1 Genre distribution

def plot_genre_distribution(df):
    counts = df["tag"].value_counts().reindex(Genre_Order)
    shares = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(Genre_Order, counts, color=Genre_Palete)
    ax.bar_label(bars, labels=[f"{n:,} ({p:.1f}%)" for n, p in zip(counts, shares)], padding=3) 
    ax.invert_yaxis() # biggest genre on top
    ax.set_xlim(0, counts.max() * 1.25) #to fit
    ax.set_xlabel("Number of songs")
    ax.set_title("Genre distribution in the training set")
    plt.tight_layout()
    plt.show()

    print(f"Imbalance ratio (largest / smallest class): {counts.max() / counts.min():.1f}x")


# 5.2 Train vs test

def plot_train_vs_test(train, test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Lyric length (log scale because of the very long texts)
    sns.kdeplot(train["word_count"], ax=axes[0], log_scale=True, color=MainColor, label="train")
    sns.kdeplot(test["word_count"], ax=axes[0], log_scale=True, color=SecondColor, ls="--", label="test")
    axes[0].set_xlabel("Words per lyric (log scale)")
    axes[0].set_title("Lyric length")
    axes[0].legend()

    # Release year (only 1950-2024 so the plot is readable)
    train_years = train.loc[train["year"].between(1950, 2024), "year"]
    test_years = test.loc[test["year"].between(1950, 2024), "year"]
    sns.kdeplot(train_years, ax=axes[1], bw_adjust=0.5, color=MainColor, label="train")
    sns.kdeplot(test_years, ax=axes[1], bw_adjust=0.5, color=SecondColor, ls="--", label="test")
    axes[1].set_xlabel("Release year (1950-2024)")
    axes[1].set_title("Release year")
    axes[1].legend()

    fig.suptitle("Train vs test: do both sets come from the same distribution?", fontweight="bold")
    plt.tight_layout()
    plt.show()


def train_test_summary(train, test):
    """Small table comparing train and test."""
    def summary(df):
        return {
            "rows": len(df),
            "median words": df["word_count"].median(),
            "median year": df["year"].median(),
            "oldest year": df["year"].min(),
            "median log1p(views)": df["log_views"].median(),
        }
    return pd.DataFrame({"train": summary(train), "test": summary(test)}).round(2)


# 5.3 Release year

def plot_release_year(df):
    print(f"Years go from {df['year'].min():.0f} to {df['year'].max():.0f} "
          f"(songs per interval in section 5.4)")
    print(f"Share of songs with year = 2015: {(df['year'] == 2015).mean() * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Songs per year, with 2015 in a different colour
    years = df.loc[df["year"].between(1960, 2024), "year"].astype(int)
    per_year = years.value_counts().sort_index()
    colors = [SecondColor if y == 2015 else MainColor for y in per_year.index]
    axes[0].bar(per_year.index, per_year.values, color=colors)
    axes[0].set_xlabel("Release year")
    axes[0].set_ylabel("Number of songs")
    axes[0].set_title("Songs per year (1960-2024), 2015 in orange")

    # Year by genre
    sns.boxplot(data=df[df["year"] >= 1950], x="year", y="tag",
                order=Genre_Order, palette=Genre_Palete, fliersize=1, ax=axes[1])
    axes[1].set_xlabel("Release year (>= 1950)")
    axes[1].set_ylabel("")
    axes[1].set_title("Release year by genre")

    plt.tight_layout()
    plt.show()


# 5.4 Genres over time (year intervals)

def plot_genres_over_time(df):
    """Number of songs per year interval and the share of each genre inside each interval."""
    counts = pd.crosstab(df["year_interval"], df["tag"])
    counts = counts.reindex(index=Year_labels, columns=Genre_Order, fill_value=0)  # show every interval
    shares = (counts.div(counts.sum(axis=1), axis=0) * 100).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 2]})

    # How many songs there are in each interval (with the number on top, the old intervals are small)
    total = counts.sum(axis=1)
    bars = axes[0].bar(total.index.astype(str), total.values, color=MainColor)
    axes[0].bar_label(bars, labels=[f"{n:,}" for n in total.values])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Number of songs")
    axes[0].set_title("Songs per interval")
    axes[0].tick_params(axis="x", rotation=45)

    # Which genres are in each interval (each bar adds up to 100%)
    shares.plot(kind="bar", stacked=True, color=Genre_Palete, width=0.8, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("% of the songs in the interval")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Genre share per interval")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return shares.round(1)


# 5.5 / 5.6 / 5.7 Boxplots by genre (views, length, lexical richness)

def plot_by_genre(df, column, ylabel, title, log_scale=False, kind="box"):
    """Box (or violin) plot of one numeric column for each genre."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if kind == "violin":
        sns.violinplot(data=df, x="tag", y=column, order=Genre_Order, palette=Genre_Palete,
                       cut=0, inner="quartile", ax=ax)
    else:
        sns.boxplot(data=df, x="tag", y=column, order=Genre_Order, palette=Genre_Palete,
                    fliersize=1, ax=ax)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def median_by_genre(df, column):
    """Median of a column for each genre, as a one-row table."""
    return df.groupby("tag")[column].median().reindex(Genre_Order).to_frame(f"median {column}").T


def add_lexical_richness(df):
    """Type-token ratio = unique words / total words (low value = repetitive lyric)."""
    tokens = df["lyrics"].fillna("").str.lower().str.findall(r"[a-z']+")
    df["n_tokens"] = tokens.str.len()
    df["ttr"] = tokens.apply(lambda t: len(set(t)) / len(t) if len(t) > 0 else np.nan)
    return df


def lexical_richness_table(df):
    # The TTR goes down when texts get longer, so we also compare songs of similar length
    similar_length = df[df["n_tokens"].between(150, 300)]
    table = pd.DataFrame({
        "median TTR (all)": df.groupby("tag")["ttr"].median(),
        "median TTR (150-300 words)": similar_length.groupby("tag")["ttr"].median(),
    })
    return table.reindex(Genre_Order).round(3).T


# 5.8 Section markers

def plot_section_markers(df):
    has_section = df["lyrics"].str.contains(Section, case=False, regex=True)
    share = (has_section.groupby(df["tag"]).mean() * 100).reindex(Genre_Order)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(Genre_Order, share, color=Genre_Palete)
    ax.bar_label(bars, fmt="%.0f%%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of songs with section markers")
    ax.set_title("Songs containing [Verse] / [Chorus] / [Hook] ... markers, by genre")
    plt.tight_layout()
    plt.show()

    n_performer = df["lyrics"].str.contains(Performer, regex=True).sum()
    print("Songs whose markers name a performer, e.g. [Verse 1: Artist]:", n_performer)


# 5.9 Data quality

def english_stopword_share(text):
    """Share of the words that are English stopwords (low value = probably not English)."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) == 0:
        return 0
    return np.mean([w in ENGLISH_STOP_WORDS for w in words])


def data_quality_issues(df, test):
    tags_per_lyric = df.groupby("lyrics")["tag"].transform("nunique")
    english_share = df["lyrics"].fillna("").apply(english_stopword_share)
    translation_accounts = r"^Genius .*(?:Transl|Tradu|Romaniz|Übersetz)"

    issues = {
        "Duplicate lyrics (extra copies)": df["lyrics"].duplicated().sum(),
        "Rows in same-lyrics groups with conflicting tags": (tags_per_lyric > 1).sum(),
        "Placeholder text (\"Tell us that you would like...\")":
            df["lyrics"].str.contains("Tell us that you would like to have the lyrics", regex=False).sum(),
        "Lyrics containing URLs": df["lyrics"].str.contains(URL, regex=True).sum(),
        "Genius translation accounts": df["artist"].str.contains(translation_accounts, regex=True).sum(),
        "Likely non-English (< 10% English stopwords)": (english_share < 0.10).sum(),
        "Very short lyrics (< 18 words)": (df["word_count"] < 18).sum(),
        "Missing title": df["title"].isna().sum(),
        "Train lyrics also present in test": df["lyrics"].isin(test["lyrics"]).sum(),
    }
    return pd.Series(issues).sort_values()


def plot_data_quality(issues, n_rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(issues.index, issues.values, color=MainColor)
    ax.bar_label(bars, labels=[f"{v:,} ({v / n_rows * 100:.2f}%)" for v in issues.values], padding=3)
    ax.set_xlim(0, issues.max() * 1.3)
    ax.set_xlabel("Rows affected in train")
    ax.set_title(f"Data-quality issues in train (n = {n_rows:,})")
    plt.tight_layout()
    plt.show()


# 5.10 Most distinctive words per genre

def song_counts_per_genre(df, min_df=20):
    """For each genre and word: in how many songs of that genre the word appears.
    binary=True so one very long, repetitive text cannot dominate a small genre."""
    vectorizer = CountVectorizer(stop_words=EDA_stopwords, binary=True, min_df=min_df,
                                 token_pattern=Word)
    X = vectorizer.fit_transform(df["lyrics"].fillna(""))
    words = vectorizer.get_feature_names_out()

    counts = {}
    for g in Genre_Order:
        rows = (df["tag"] == g).values
        counts[g] = np.asarray(X[rows].sum(axis=0)).ravel()
    return pd.DataFrame(counts, index=words)


def log_odds_z(counts, genre):
    """Weighted log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008).
    Compares one genre against all the others; a high z-score means the word is much
    more common in that genre. The prior stops very rare words from getting huge scores."""
    y_genre = counts[genre]
    y_rest = counts.drop(columns=genre).sum(axis=1)
    prior = counts.sum(axis=1) + 1
    a0 = prior.sum()

    log_odds_genre = np.log((y_genre + prior) / (y_genre.sum() + a0 - y_genre - prior))
    log_odds_rest = np.log((y_rest + prior) / (y_rest.sum() + a0 - y_rest - prior))
    variance = 1 / (y_genre + prior) + 1 / (y_rest + prior)
    return (log_odds_genre - log_odds_rest) / np.sqrt(variance)


def plot_distinctive_words(df, top_n=15):
    counts = song_counts_per_genre(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for genre, ax in zip(Genre_Order, axes.flat):
        top_words = log_odds_z(counts, genre).sort_values().tail(top_n)
        ax.barh(top_words.index, top_words.values, color=Genre_Colors[genre])
        ax.set_title(genre)
        ax.set_xlabel("log-odds z-score")
    fig.suptitle("Most distinctive words per genre (raw lyrics, stopwords removed)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# 5.11 Vocabulary similarity between genres

def plot_genre_similarity(df):
    vectorizer = TfidfVectorizer(stop_words=EDA_stopwords, min_df=20, max_features=50_000,
                                 sublinear_tf=True, token_pattern=Word)
    X = vectorizer.fit_transform(df["lyrics"].fillna(""))

    # Average TF-IDF vector of each genre, then cosine similarity between them
    mean_vectors = [np.asarray(X[(df["tag"] == g).values].mean(axis=0)).ravel() for g in Genre_Order]
    sim = pd.DataFrame(cosine_similarity(mean_vectors), index=Genre_Order, columns=Genre_Order)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="Blues", vmin=0.6, vmax=1, square=True,
                cbar_kws={"label": "Cosine similarity"}, ax=ax)
    ax.set_title("Vocabulary similarity between genres\n(cosine similarity of mean TF-IDF vectors)")
    plt.tight_layout()
    plt.show()


# 5.12 Artists split across genres

def plot_split_artists(df, min_songs=15, n_artists=12):
    # Songs per artist and genre (Genius translation accounts are not real artists)
    real_artists = df[~df["artist"].str.startswith("Genius ")]
    artist_tags = pd.crosstab(real_artists["artist"], real_artists["tag"])[Genre_Order]

    multi = artist_tags[(artist_tags > 0).sum(axis=1) > 1]
    n_songs = multi.values.sum()
    print(f"Artists with songs under more than one tag: {len(multi):,} "
          f"({n_songs:,} songs, {n_songs / len(df) * 100:.1f}% of train)")

    # Most evenly split artists = the smallest share for their main genre
    big = multi[multi.sum(axis=1) >= min_songs]
    main_genre_share = big.max(axis=1) / big.sum(axis=1)
    top = big.loc[main_genre_share.sort_values().index[:n_artists]]
    top = top.loc[top.sum(axis=1).sort_values().index]   # longest bar on top

    ax = top.plot(kind="barh", stacked=True, color=Genre_Palete, figsize=(10, 6), width=0.7)
    ax.set_xlabel("Number of songs")
    ax.set_ylabel("")
    ax.set_title(f"{n_artists} most evenly split artists (>= {min_songs} songs each)")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.show()
