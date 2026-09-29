# Collaborative Filtering Movie Recommender

A single-page web application that recommends movies using **collaborative filtering**.
You pick a user from the MovieLens 100K dataset and the app produces **two Top-5 lists
computed two different ways** — *User-Based CF* (what similar people liked) and *Item-Based
CF* (movies similar to the ones this user already rated) — so the two approaches can be
compared side by side for the same user.

Built with vanilla HTML, CSS, and JavaScript — no frameworks, no build step.

## Features

- **Side-by-side comparison** — the two Top-5 lists are always shown together for the same
  user, so their overlap and their score spread are directly comparable.
- **Pick the Item-Based seed** — a second dropdown, `#movie-select`, lists the movies the
  selected user has rated, best-rated first. The chosen movie anchors the Item-Based panel:
  it decides which of that user's rated movies are allowed to contribute similarity, so the
  list genuinely changes when you change it. The active user and the chosen title are shown
  in the page and logged to the console.
- **Co-rated cosine similarity** — both approaches use the *same* similarity function,
  computed only over movies both parties actually rated.
- **User-Based CF** — the 20 most similar users (positive similarity, at least 5 co-rated
  movies) are kept, and every unseen movie is scored as the similarity-weighted average
  of those neighbours' ratings.
- **Item-Based CF** — an item × item similarity table is precomputed once at load, then
  every unseen movie is scored by aggregating similarities from the movies the user *did*
  rate, weighted by the rating given.
- **Instant queries** — the expensive item × item table is built once in chunked batches
  (never freezing the page), so each button click answers in roughly 20 ms.
- **Graceful degradation** — users with almost no history, or IDs outside the dataset,
  return a clear explanation instead of an error or an empty box.
- **Responsive layout** — the two panels stack on narrow screens.

## Project Structure

```
├── index.html   # Page structure (user + movie dropdowns, two result panels)
├── style.css    # Layout and styling
├── data.js      # Data module: loads & parses u.item and u.data
├── script.js    # Similarity, both CF algorithms, and UI rendering
├── u.item       # Movie metadata (id, title, 19 genre flags)
└── u.data       # 100,000 ratings (userID, itemID, rating, timestamp)
```

## Getting Started

### Try it online

The app is deployed with GitHub Pages:

**<https://pollyiva.github.io/Collaborative-Filtering-Movie-Recommender/>**

### Prerequisites

A modern browser with `fetch()` support. The app reads `u.item` and `u.data` via `fetch()`,
which requires the page to be served over HTTP — opening `index.html` directly from the
filesystem will not work, since browsers block `file://` requests.

## Usage

1. Pick a user (1–943) from the dropdown.
2. Click **Get Recommendations**.
3. The two Top-5 lists appear side by side, with the score each one produced.

## How It Works

1. **Data loading** (`data.js`): `loadData()` fetches `u.item` and `u.data` and produces:

   - `movies` — `{ id, title, genres }`, where `genres` comes from the 18 named genre flags.
   - `ratings` — `{ userId, itemId, rating, timestamp }`.
   - `ratingMatrix` / `ratedMask` — the user × movie matrix plus a parallel boolean mask,
     so an explicit `0` rating can never be confused with "not rated".

2. **Similarity** (`script.js`): `cosineSimilarity(a, b)` scores two rating vectors using
   `dot(a, b) / (‖a‖ · ‖b‖)`, where the dot product and both norms are accumulated **only
   over indices where both vectors are non-zero**. Pairs with no overlap return `0`.

3. **User-Based CF**: every other user is compared to the active user, the 20 most similar
   are kept, and each unseen movie's score is `Σ(sim × rating) / Σ(sim)` over those
   neighbours who rated it.

4. **Item-Based CF**: `buildItemSimilarities()` walks a sparse user → item index and, for
   every pair of movies a user rated together, accumulates the cosine numerator and both
   norms into a full item × item table. Doing it this way costs roughly 10 million pair
   updates instead of the ~1.17 billion a naive movie × movie × user triple loop would need.
   Each unseen movie is then scored as the similarity-weighted average of the ratings for the
   seed's nearest rated movies. With no seed selected, every rated movie contributes, which
   is the plain basket aggregation; with one, the ten rated movies closest to the seed
   contribute instead.

5. **Output**: both lists are ranked, truncated to 5, and rendered with their scores.

## Missing Values

The rating matrix is **93.7% empty**, so the missing-value strategy matters more than the
formula. A movie a user has not seen is *skipped*, not treated as a rating of zero —
otherwise every unrated movie would look like a strong dislike and the similarities would be
meaningless. The same co-rated-only rule is applied in every function that touches a vector.

## A Note on the Scores

User-Based scores come out very close to `5.00` in practice. This is expected rather than
broken: cosine similarity is not offset-corrected, so a neighbour who rated the shared
movies `5/5` has similarity `1.0` with the active user, and the similarity-weighted average
then reproduces their `5`s almost exactly. Item-Based scores spread across a much wider
range for the same reason — aggregating over many similar items mixes in a broader set of
ratings. Mean-centering (e.g. Pearson correlation) would give more discriminative
User-Based rankings.

## Analysis

Everything below is measured on the actual dataset shipped with this project (943 users,
1,682 movies, 100,000 ratings = 1,586,726 cells, **93.70% empty**), not estimated.

### Comparing users (columns) vs items (rows)

`ratingMatrix` has the shape `(numUsers + 1) × (numMovies + 1)` as the brief requires, so in
the raw matrix **users are rows and items are columns**:

- **User-Based CF** compares two *user rows* — `cosineSimilarity(activeRow, ratingMatrix[other])`.
- **Item-Based CF** compares two *item columns*, but the result is cached transposed in the
  precomputed `itemSim` table, in which **items are rows** (`itemSim[i * width + j]`). A
  query therefore reduces to a row-wise table lookup.

That transposition is not cosmetic: it is what lets a single flat `Float64Array` answer
"how similar are items *i* and *j*?" in O(1) instead of a strided column walk.

### User-Based vs Item-Based: quality and cost

| | User-Based | Item-Based |
|---|---|---|
| Pairs to precompute | **444,153** (943²/2) | **1,413,721** (1,682²/2) |
| Pairs with <5 co-rated | 26.0% | **70.1%** |
| Pairs with zero overlap | 3.3% | **30.3%** |
| Query cost (measured) | 5.59 ms | **1.54 ms** |

**Cost depends on which population is larger.** Precompute grows with the *square* of the
entity count: O(|U|²) for users versus O(|I|²) for items. Here items outnumber users
1.78× (1,682 vs 943), so the item-item table is **3.18× larger** to build — the more
expensive precompute. But that cost is paid **once**, whereas the user-based query above
re-scores against 20 neighbours for every candidate on every click, which is why it is
**3.6× slower per query**.

The general rule follows from the arithmetic: user-based wins when |U| < |I| (fewer pairs to
precompute), and item-based wins decisively when |U| >> |I|. A service with millions of
users and a few thousand items — the common case — cannot build a user-user table at all
(it would be O(10¹²) pairs), so industry systems are item-based or matrix-factorization by
necessity, not preference.

**Quality is the more interesting axis, and the two lists barely agree.** Across 236 users
the User-Based and Item-Based Top-5 lists share a mean of **0.00 titles**; for **99.6% of
users the two lists are completely disjoint**. Same user, same data, same similarity
function — yet almost no overlap. This is a property of the rating scale, not a bug:
User-Based scores saturate at 5.00 (see above) and surface whatever the user's nearest
neighbours loved, whereas Item-Based propagates outward from the seed's similarity
neighbourhood. The two answers are answering subtly different questions, which is precisely
why the assignment asks for them side by side.

**Reliability of the underlying evidence differs by direction.** Users are dense (median 65
ratings, minimum 20) while items are sparse (median 27, and **333 items have fewer than 5**).
Consequently item-item similarity is far more often estimated from almost no data — 70.1%
of item pairs have under 5 co-rated ratings and 30.3% share none at all. On this dataset
User-Based CF is the statistically better-evidenced of the two, while Item-Based CF is the
faster and the one that scales.

### Missing-Value Strategy Trade-off

The brief offers three strategies. This project uses **co-rated items only** (missing
entries skipped, never imputed), stated at the top of `script.js` per §6 of the prompt.
Matrix factorization was analysed but **not** implemented.

| Strategy | Simplicity | Bias | Cost | Behaviour under sparsity |
|---|---|---|---|---|
| **Co-rated only** (used) | Highest — one loop, no extra state | Lowest — invents no data | Lowest | High variance: a similarity from 2 shared films is noise, not signal |
| **Mean imputation** | Low — fill gaps once, then treat as dense | **Highest** — asserts "unrated = average taste", which is false, and flatters users with few ratings | Low at query time, but the matrix is 1.6M cells and must be rebuilt | Hides the problem: every pair becomes computable, including pairs that should not be |
| **Weighted by common ratings** | Medium — carries a count per pair | Low | Medium — an extra counter per accumulation | Shrinks unreliable pairs toward zero instead of trusting them blindly |
| **Matrix factorization** | Lowest — it is a trained model, not a similarity | Low, but **popularity bias** returns via the global term | **Highest** — training loop, epochs, regularisation, tuning | **Best** — low rank shares strength, so a 1-rating item still gets a usable latent vector |

Two consequences worth stating plainly:

- **Mean imputation is the trap here.** Filling a 93.7%-empty matrix with row means makes
  every unseen (user, item) pair look like mild indifference, so a user who simply has not
  seen a film becomes indistinguishable from a user who mildly disliked it. It also biases
  toward blockbusters: heavily-rated items have tightly concentrated column means, so they
  win every comparison. Since one of this course's stated business goals is *catalog
  discovery* of long-tail titles, imputation works directly against it.
- **This implementation uses a threshold, not a weight.** `MIN_CO_RATED = 5` discards pairs
  with too little overlap outright. That is a deliberately crude, binary cousin of the
  "weighted by common ratings" option — it removes the noise instead of shrinking it. A
  continuous shrinkage weight (divide by √n_co-rated, say) would keep those pairs and let
  evidence speak proportionally, at the cost of carrying counts through every accumulation.

### Cold Start and Sparsity

**A new user cannot be served.** Collaborative filtering has no content to work from: with
zero ratings there is no vector, therefore no similar users, and no similarity-weighted
prediction can be formed. The item-item table does exist, but it is not personalised — there
is nothing to anchor it to. The failure is structural, not a matter of tuning.

**A new item can never be recommended by Item-Based CF.** An unrated movie has an all-zero
column, so its row in `itemSim` is identically zero and it has no similarity to anything. It
cannot surface through User-Based CF either, since no neighbour has rated it. Such items are
invisible to the entire system.

**Too little evidence looks like agreement.** This is the subtlest failure, and it is
quantified above: 30.3% of item pairs share no ratings at all, so the cosine denominator is
zero and the similarity is undefined — those pairs are dropped, silently shrinking the
candidate pool. Of the pairs that *are* defined, 70.1% rest on fewer than 5 shared ratings.
A cosine computed from two shared films can easily be exactly 1.0 (two users who both gave
one film a 4 and nothing else), which is why naive co-rated cosine lets noise masquerade as
a perfect match. That is the specific reason `MIN_CO_RATED` exists, and why the panel states
the number of neighbours used rather than presenting a score as if it were trustworthy.

Matrix factorization mitigates the *item* cold start by learning a low-rank structure in
which nearby items borrow latent factors from each other, but it does not fix *user* cold
start: a new user still contributes no factors. Both cases ultimately fall back to
content-based signals (genres, year, cast), explicit onboarding, or demographics.

## Data Source

The `u.item` and `u.data` files come from the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/):

- 1,682 movies with genre flags (Action, Adventure, …, Western)
- 100,000 ratings from 943 users on a 1–5 scale

> The files are stored in **Latin-1**, not UTF-8. The parser decodes them with
> `TextDecoder('latin1')` so accented titles such as *Les Misérables* survive intact.

## License

This project is for educational purposes. The MovieLens dataset is made available by
GroupLens for research and educational use — see the
[MovieLens terms](https://grouplens.org/datasets/movielens/).
