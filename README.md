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
