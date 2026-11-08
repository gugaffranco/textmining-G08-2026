# Data quality notes: `train.csv`

Notes from exploring `datasets/train.csv` (134,967 rows; columns `title, tag, artist, year, views, features, lyrics, id`).
The goal is to decide **what to clean, what to drop and what to leave alone** before modelling.

> **Main takeaway:** the labels are noisy but *mostly* fine. Fewer than 1% of songs look clearly mislabelled.
> The bigger problems are formatting artifacts, non-lyric pages and genres that genuinely overlap (pop and rock).
> Don't over-clean: `test.csv` comes from the same source and has the same quirks.

---

## 0. Quick facts

| | |
|---|---|
| Rows | 134,967 train / 33,742 test |
| Missing values | only 2 in `title` |
| Duplicate `id` | 0 |
| Duplicate title + artist | 0 |
| Train/test `id` overlap | 0 |

**Class balance** (very unbalanced; use macro-F1 and/or class weights):

| tag | share |
|---|---|
| pop | 41.3% |
| rap | 28.6% |
| rock | 18.8% |
| rb | 4.6% |
| misc | 4.2% |
| country | 2.6% |

**Lyric length (words) per genre.** Length on its own is a strong signal:

| tag | median | mean | max |
|---|---|---|---|
| rap | 415 | 437 | 9,831 |
| rb | 291 | 309 | 2,293 |
| misc | 258 | **974** | 18,221 |
| country | 217 | 232 | 7,092 |
| pop | 205 | 227 | 3,775 |
| rock | 188 | 202 | 2,246 |

---

## 1. Mislabelled songs (rap tagged as another genre, and the reverse)

### How we checked
1. **Classifier check.** We trained a TF-IDF + linear classifier with 5-fold cross-validation, so every song is scored by a model that **never saw it during training**.
   A song is flagged when the model is **> 90% sure** of another genre **and** gives the actual tag **< 3%**.
2. **Artist majority.** For artists with ≥ 5 songs where ≥ 80% share one tag, we flagged the songs that don't follow the majority.
3. **Rap-slang density.** We counted the share of words like *nigga, bitch, hoes, glock, opps, drip, racks, bars, flow, mic…*
   Median per 100 words: rap 0.76 and every other genre ≈ 0.

### Results
We ran two classifiers:

| model | macro-F1 (5-fold) | accuracy | songs flagged |
|---|---|---|---|
| fast: unigrams, 50k features, SGD | 0.514 | 0.629 | 950 |
| big: unigrams + bigrams, 300k features, logistic regression | **0.549** | **0.646** | 902 |

**469 songs are flagged by both models.** This is the most reliable list of mislabels (~0.35% of the data).
Rows = actual tag, columns = what the lyrics look like:

| tag ↓ / looks like → | country | misc | pop | rap | rb | rock |
|---|---|---|---|---|---|---|
| country | – | 1 | 0 | 7 | 0 | 1 |
| misc | 0 | – | 3 | **52** | 0 | 1 |
| pop | 1 | 2 | – | **283** | 1 | 11 |
| rap | 0 | 4 | 0 | – | 0 | 1 |
| rb | 0 | 1 | 2 | **66** | – | 0 |
| rock | 0 | 0 | 1 | 31 | 0 | – |

So the clearest error, by far, is **rap songs tagged as something else (≈ 440 of the 469)**, mostly as pop.
Pop/rock disagreements are common in the fast model (127 rock songs flagged as "pop") but **mostly disappear in the big model**.
That means they're overlap between the genres, not wrong labels.

Results from the big model alone (902 flagged):

| tag ↓ / looks like → | country | misc | pop | rap | rb | rock |
|---|---|---|---|---|---|---|
| country | – | 4 | 4 | 11 | 0 | 4 |
| misc | 0 | – | 5 | **82** | 0 | 3 |
| pop | 21 | 23 | – | **454** | 29 | 19 |
| rap | 2 | **30** | 0 | – | 17 | 4 |
| rb | 2 | 2 | 6 | **115** | – | 0 |
| rock | 6 | 4 | 7 | 43 | 5 | – |

### Examples: rap songs NOT tagged rap
| title | artist | tag | why it's rap |
|---|---|---|---|
| Way I Am | Snoop Dogg | pop | *"I'm the most underrated rapper of all time / So why do you accept those that bite my rhymes"* |
| Say It To My Face | Meek Mill | pop | *"Before I eat an MC I say my grace…"* |
| Crank It Up | Webbie | pop | Southern rap, full of slang |
| 25 To Life | Xzibit | pop | rapper; the artist's songs are mostly tagged rap |
| In Too Deep | Kool G Rap | pop | rapper |
| Air It Out | Mobb Deep | pop | rap duo |
| U aint $$$tt | DMX | pop | rapper |
| Rarris and Rovers | Tink | rb | rap verses |
| Useless | BONES | rock | emo-rap / cloud rap, *"[Verse] They say that they fuck with you…"* |
| Welcome to california / Grave Yard | The Game | pop | rapper |
| Takin Pictures | Fabolous | pop | rapper |
| Exodus | Ice-T | pop | rapper |
| Up In Tha Hood | Hot Boys | pop | rap group |
| Chain Reaction | 7L & Esoteric | pop | rap duo |
| Calicoe vs. Aye Verb / Murda Mook vs. Tay Roc | URLtv | misc | **rap battles** (in the past they were filed as misc, not rap) |
| The LOX Freestyle Funkmaster Flex | The LOX | misc | radio freestyle |

Also: **295 songs** are by artists who are ≥ 80% rap but are tagged something else, and the classifier agrees "rap" for **177** of them.

### Examples: NOT rap but tagged rap
| title | artist | tag | what it actually is |
|---|---|---|---|
| Chapter 2: Variation Under Nature | Charles Darwin | rap | *On the Origin of Species* (book chapter) |
| The Bean-Field | Henry David Thoreau | rap | *Walden* (book chapter) |
| A Statement on Plan B One-Step… | U.S. Dept of Health & Human Services | rap | government press statement |
| Do You Love Me? | Ariana Grande | rap | pop song |
| You I | Crystal Fighters | rap | indie pop |
| Intuition | Keri Hilson | rap | R&B |
| Horizon | I.M (MONSTA X) | rap | K-pop / R&B |

**The rap tag also contains non-song pages** (about 30–40 rows) that should really be misc:

| title | artist | what it is |
|---|---|---|
| Week 7 Reading journal | Alejandro Rodriguez / Hoa Huynh | student homework |
| How to study math: 6 practical tips | Emma Stone (not the actress) | blog/SEO article |
| Eight footballers say we're gay but keep quiet… | The Guardian | newspaper article |
| Fat Trel Reportedly Signed To M.M.G. | Hip Hop DX | music news article |
| A Modest Proposal to Ensure Medicare's Long-Term… | Ryan Morgan | essay |
| SAFE ABORTION CLINIC CALL DR AMINAH @07176… | AMINAH | **spam ad with a phone number** |
| An Open Letter To President Obama | Russell Simmons | letter |
| Frank Ocean - 'undocumented RARE.' | Blonded.blog | fan blog post |

These pages are probably tagged rap because they were posted in Genius's rap section (e.g. hip-hop news) or by users who mostly post rap.

### Pattern: year 2015
**21%** of the rap songs tagged as other genres have `year = 2015`, against **9%** in the whole dataset.
2015 may be a default value used when the scraper didn't know the date, and those rows also seem to get worse tags.

---

## 2. Genre is assigned per artist, not per song

- **6,575 artists** have songs under more than one tag (31,814 songs).
- Well-known artists are split almost in half:

| artist | split |
|---|---|
| Elvis Presley | country 23 / rock 13 / pop 5 |
| Taylor Swift | country 13 / pop 9 / misc 2 |
| The Cure | pop 14 / rock 13 |
| Billy Joel | pop 12 / rock 11 |
| Prince | pop 13 / rock 7 / rb 5 |
| Aretha Franklin | rb 15 / pop 7 / rock 2 |
| Tory Lanez | rap 16 / rb 7 |

- **Pop and rock are the noisiest pair.** About 10,600 pop songs "look like" rock and about 10,500 rock songs "look like" pop.
  This is real genre overlap, not a cleaning issue. Expect pop, rock, rb and country to score worse than rap and misc.

---

## 3. Non-lyric content

### 3.1 `misc` is basically "not a song"
The top `misc` artists are writers and institutions, not musicians:

> Emily Dickinson (74), Abraham Lincoln (59), Holy Bible (KJV) (50), Charles Dickens (48), William Shakespeare (48), Thomas Hardy, Leo Tolstoy, Walt Whitman, Mark Twain, Nietzsche, Karl Marx, The SCP Foundation…

Example `misc` titles: *"To Kill a Mockingbird Chapter 3"*, *"North and South Chap. 18"*, *"2015 NBA Playoffs Day 1 Recap"*, *"Kane WWE Profile"*, *"Spider Extermination in Norfolk VA"* (spam), *"Career research project: High School Teacher"* (school work), *"Inglourious Basterds - The Jew Hunter Speech"*.

**➡ Keep it.** The label is consistent ("not music"), and the test set almost surely has the same kind of rows. Deleting them would make our test predictions worse.

### 3.2 Translations
**606 rows** come from Genius translation accounts instead of the real artist:

| account | rows |
|---|---|
| Genius English Translations | 555 |
| Genius Traductions Françaises | 21 |
| Genius Romanizations | 13 |
| Genius Brasil Traduções | 9 |
| others | 8 |

These are real songs (pop 321, rap 192, rock 57, rb 35), but the text is a translation and the `artist` field is wrong.
**➡ Group decision:** keep them (the genre is valid) or test with and without them.

### 3.3 Empty or broken lyrics (drop these)
| title | lyrics |
|---|---|
| Activated part 1 - Origins of a Rap Acrobat | `This page has been moved to here... http://genius.com/...` |
| Alison Ritz … Minimum Wage Debate Project … | just a link to an Atlantic article |
| (tour page) | `Date To Be Announced01.1102.1103.11…` |

- **154 lyrics contain URLs.** Remove them with a regex.
- **55 short entries are mostly instrumental markers**, e.g.
  `"[Instrumental]\n\n[Instrumental buildup]\n*inaudible high-pitched voice*\n[Beat Drop]\nInsane inflame"`.

### 3.4 Not in English
- **About 107 songs** have very few common English words (misc 39, pop 31, rock 18, rap 14, rb 5).
- 5 songs are mostly in a non-Latin script.
- Some are credit lists or tracklists rather than lyrics (e.g. *"Velvet Portraits Credits"*).

---

## 4. Duplicates and test-set overlap

There are **343 rows with exactly the same lyrics as another row** (165 groups; 178 rows are "extra copies").
They fall into three types:

### 4.1 Placeholder text: DROP (11 rows in train, 6 in test)
This isn't really a duplicate problem: it's **fake lyrics that Genius shows when the real lyrics are missing**:

> *"Tell us that you would like to have the lyrics of this song. Then we'll make it our highest priority"*

| title | artist | tag |
|---|---|---|
| If I Only Had Time | Jason Donovan | pop |
| Camber Sands | Fatboy Slim | pop |
| How Do We Get Back to Love | Julia Michaels | pop |
| Life At Sea | Wiley | pop |
| JJ HATER | AzTBM | rap |
| … (11 in total, 10 pop) | | |

Drop them from train. The 6 in test can't be dropped, but there's no text to learn from anyway; predicting `pop` (the majority class) for them is fine.

### 4.2 Same text, different tags: 13 groups (26 rows)
Almost all are **the same song in two versions** (live, acoustic, first/second pressing, cover), labelled differently:

| title | artist | tag |
|---|---|---|
| Fingertips Part 1 First Pressing | Stevie Wonder | pop |
| Fingertips Part 1 Second Pressing | Stevie Wonder | rb |
| Honey Honey | Feist | rock |
| Honey Honey BBC session | Feist | pop |
| Wrong | Evergrey | rock |
| Wrong live unplugged | Evergrey | pop |
| Gin House Blues | The Animals | rock |
| Gin House Blues | Eric Burdon & The Animals | pop |
| Angelina | Caiphus Semenya | pop |
| Angelina Live | Caiphus Semenya | rb |
| Twang | Mason Ramsey | country |
| Twang | Primrose Ripper | pop |
| Love Yourself | The Maine | rock |
| Love Yourself | Nicole Milik | pop |

Almost every case is pop against rock/rb, which is the same overlap as in section 2.
**➡ Simplest option: drop all 26 rows.** It's so few that it doesn't matter, and we avoid guessing which tag is right.
(Alternative: keep the copy with the most `views`, which is usually the original release.)

### 4.3 Same text, same tag: 152 groups (317 rows)
These are harmless for the labels, but they are **copies**: remixes, radio edits, instrumentals or the same poem posted by two users.

| group | tag |
|---|---|
| "Damage Ive Done Moby Sad Gospel mix" / "…Sound/Bisquit vocal mix" (The Heads) | pop |
| "Communication Remix" / "Communication Long Remix" (The Power Station) | pop |
| "Switch It Off" / "Switch It Off - Radio Edit" (Redlight) | pop |
| "Virtual Friend" / "Virtual Friend Acoustic Version" (Armin van Buuren) | pop |
| George Orwell's "Animal Farm: Chapter 6" posted twice (Mr. Allen) | misc |
| "Bright Star" (John Keats) / "John Keats's 'Bright Star'" (MsRamsey) | misc |
| MLK's *"I Have a Dream"* speech (3 copies) | misc |
| "Othello" summary by a Genius user (4 copies) | misc |

**➡ Keep one copy per group** (`df.drop_duplicates(subset="lyrics")`).
**Why this matters:** with cross-validation, one copy can land in the training fold and the other in the validation fold.
The model then "predicts" a text it already saw, and the validation score comes out too optimistic.

### 4.4 Overlap with test
- **90 train rows** have lyrics that appear word for word in `test.csv` (pop 64, misc 17, rock 6, country 2, rap 1).
- Nothing to remove here; just be aware of it.

**Order of operations:** (1) drop placeholders → (2) drop conflicting groups → (3) `drop_duplicates(subset="lyrics")`.

---

## 5. Formatting: section headers

Genius lyrics often contain markers like `[Verse 1: Artist]`, `[Hook]`, `[Chorus]`, `[Intro]`, `*inaudible*`, `(x2)`.
How often a song has them depends heavily on the genre:

| tag | has `[Verse]`/`[Chorus]`… |
|---|---|
| rap | 65% |
| rb | 60% |
| country | 50% |
| rock | 36% |
| pop | 29% |
| misc | 8% |

If we keep them, the model learns *"has brackets → rap/rb"*, which describes how Genius users format pages, not the lyrics.
**➡ Remove `[ … ]` blocks during preprocessing**, e.g. `re.sub(r"\[.*?\]", " ", text)`, or keep them as a separate feature only if we decide to on purpose.
They also include artist names (`[Verse 2: Lil Wayne]`), which leaks information about the artist.

Titles carry hints too:

| word in title | tags |
|---|---|
| Freestyle | rap 748 / everything else 52 |
| Cypher | rap 58 / 0 |
| Rap Battle | rap 21 / 0 |
| Chapter | misc 159 / rest 38 |
| Translation | pop 320, rap 174, rock 56… |

---

## 5b. Rap battles: KEEP them (as rap)

Real battle content comes from leagues and channels like URLtv, King of the Dot, Grind Time Now, Don't Flop, Scribble Jam, Epic Rap Battles of History and VideoGameRapBattles.

| | train | test |
|---|---|---|
| battle rows | **89** | **28** |
| tagged rap | 86 | ? |
| tagged misc | 3 | ? |

The 3 tagged misc: *"Calicoe vs. Aye Verb"* and *"Murda Mook vs. Tay Roc"* (URLtv), and *"Imagine - ADD Presents: Dialogue with Black Ice"* (Black Ice, a spoken-word piece).

**Why keep them:**
- **97% are already tagged rap**, so the label is consistent.
- Battles are rap: rhymes, slang, insults, wordplay. They teach the model real rap vocabulary.
- **The test set has ~28 battles too.** If we delete them from train, the model has never seen one and could guess misc for them.
- They are long (median 314 words) but not extreme. Rap songs in general are already the longest genre.

**Optional:** relabel the 2 URLtv battles tagged misc to rap, or just leave them; 2 rows don't change anything.

⚠️ A simple filter like `title.contains("battle|vs")` is **wrong**. It also matches *"The Battle of Life Chap. 2"* (Dickens), *"When The Battle Is Over"* (gospel), *"Me vs Maradona vs Elvis"* (Brand New, rock) and Godsmack/Disturbed songs.
Filter by battle-league artist names instead if you ever need to find them.

---

## 5c. Book words ("chapter", "act", "poem"…): DON'T remove them

The question was whether to delete words that show a text is a book, like *chapter*.
**No: these words are useful signal, not noise.**

| word | rows in train (title / lyrics) | rows in test (title / lyrics) | tags of the train rows whose lyrics contain it |
|---|---|---|---|
| `Chap.` | 565 / 10 | **158** / 5 | misc 9, rap 1 |
| `chapter` | 191 / 676 | 43 / 145 | misc 250, **rap 242**, pop 110, rock 56 |
| `act I/II/…` | 120 / 214 | 40 / 50 | rap 88, misc 78 |
| `poem` | 46 / 303 | 12 / 66 | misc 110, rap 106 |
| `scene` | 167 / 3,032 | 47 / 869 | rap 1,555, pop 708 |

**Why keep them:**
1. **`misc` = "not a song" is a real class we have to predict.** The test set has the same books: 158 test titles contain `Chap.`.
   Words like *chapter* are exactly what lets the model recognise misc. Removing them makes misc harder to predict, and misc is one of our better classes (F1 ≈ 0.62).
2. **They aren't only book words.** *"chapter"* appears in 242 rap lyrics (*"new chapter in my life"*) and *"scene"* mostly in rap.
   A manual word list would remove real lyric words too.
3. TF-IDF already gives these words the right weight. Rare, class-specific words get high IDF, so we don't need to decide by hand.

**What we *should* remove** is formatting that has nothing to do with genre (section 5): `[Verse 1: X]`, `[Chorus]`, `*inaudible*`, URLs.
Book words are content; the bracket markers are formatting.
*Rule of thumb: remove something only if it's (a) formatting rather than content, or (b) wouldn't be available or the same in test.*

---

## 6. `year` and `views`

- **45 rows have `year = 1`**, plus other impossible values such as 3, 4, 6, 14, 17, 22, 69 and 210. These are missing values written as a placeholder; the lyrics are fine.
- **Old years can be real**: Darwin (1859), Thoreau (1854) and Shakespeare (1590s) are all `misc` and correctly dated.
- ⚠️ Our current rule ("drop every row with year < 1600") **deletes good text because of one bad metadata field**.
  It's better to set impossible years to `NaN` and keep the row, since we classify from the lyrics anyway.
- `views` is very skewed (median 89, mean 3,413, max 3.6M). Use `log1p(views)` if we use it.
- `features` is not empty (`!= "{}"`) in 17% of rows. It's a set-like string, e.g. `{"Corey J."}`, and needs parsing.

---

## 7. Recommended cleaning plan

| step | action | why |
|---|---|---|
| 1 | Drop the ~4 empty, URL-only or placeholder lyrics | no content |
| 2 | Remove URLs inside lyrics | noise |
| 3 | Drop the 11 "Tell us that you would like to have the lyrics…" placeholders; drop the 13 conflicting-tag groups (26 rows); then `drop_duplicates(subset="lyrics")` (see §4) | fake text, contradictory labels, CV leakage |
| 4 | Remove `[Verse]`/`[Chorus]`/`*…*` markers | formatting habit, leaks artist names |
| 5 | Set impossible years to `NaN` instead of dropping the rows | keeps good lyrics |
| 6 | **Keep `misc`** non-song texts | consistent class, also in test |
| 7 | Decide on translations (keep, or test dropping them) | trade-off |
| 8 | **Don't mass-relabel** with a model. At most drop the 469 songs flagged by both models (Darwin tagged rap, Snoop tagged pop, homework tagged rap…), and compare scores with and without them | relabelling with a model is circular; test has the same noise |
| 9 | Use class weights and macro-F1 | strong class imbalance |
| 10 | In the report, describe the label noise (rap tagged pop, split artists) | explains the weaker pop/rock/rb/country scores |

---

### Reproducing the suspect list
The classifier output (one row per song: `tag`, `pred`, `pmax`, `p_given`, `rapdens`, `art_maj`, `art_share`, `art_n`) was saved to a CSV.
Suspects = `pred != tag & pmax > 0.9 & p_given < 0.03`.
Section 1 uses both models (the fast SGD one and the big logistic-regression one); "flagged by both" = the 469 songs in both suspect lists.
Section 2's pop/rock numbers come from the fast model; the big model gives a similar picture (≈ 10,100 pop→rock and 13,500 rock→pop "confusions"), which confirms the overlap between the genres.

**Baseline to beat:** a plain TF-IDF (1–2-grams) + logistic regression already reaches **macro-F1 ≈ 0.55 / accuracy ≈ 0.65**.
Per class F1: rap 0.87, pop 0.63, misc 0.62, rock 0.49, country 0.36, rb 0.33.
