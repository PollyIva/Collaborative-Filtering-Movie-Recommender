# Collaborative Filtering Movie Recommender

<p align="center">
  <a href="https://pollyiva.github.io/Collaborative-Filtering-Movie-Recommender/">
    <img src="https://img.shields.io/badge/Open%20the%20app-2ea44f?style=for-the-badge" alt="Open the app">
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/PollyIva/Collaborative-Filtering-Movie-Recommender/blob/master/Polina_Ivanilova_a03_report.pdf">
    <img src="https://img.shields.io/badge/Report%20(PDF)-c0392b?style=for-the-badge" alt="Report (PDF)">
  </a>
</p>

MovieLens 100K in vanilla HTML, CSS and JavaScript — no framework, no build step.
Pick a user and the app returns **two Top-5 lists for that same user**:
*User-Based CF* (what similar people rated) next to *Item-Based CF* (what resembles
the films they already rated), so the two can be compared directly.

## Quick start

**Online** — <https://pollyiva.github.io/Collaborative-Filtering-Movie-Recommender/>

**Locally** — the app reads `u.data` and `u.item` with `fetch()`, which browsers
block on `file://`, so serve the folder instead of opening `index.html` directly:

```bash
python -m http.server 8000     # then open http://localhost:8000
```

## Using it

1. Pick a user (1–943).
2. Optionally pick an **Item-Based seed** — the dropdown lists only films that user
   rated, best first. The seed decides which of their rated films are allowed to
   contribute similarity, so the Item-Based list really changes when it changes.
   Leave it alone and the app anchors on that user's highest-rated film.
3. Press **Get Recommendations**. Both panels fill in.

## How it works

- `loadData()` parses `u.item` (Latin-1) and `u.data` into a dense
  `(numUsers+1) × (numMovies+1)` `Float32Array`, a parallel `Uint8Array` mask, and
  sparse `userItems` / `itemUsers` indexes. `0` unambiguously means *not rated*.
- `cosineSimilarity(a, b)` accumulates the dot product and both norms **only over
  positions both vectors rated**. A pair with no overlap returns `0`.
- **User-Based** keeps the 20 nearest users and scores each unseen film as
  `Σ(sim × rating) / Σ(sim)`.
- **Item-Based** precomputes all 1,413,721 film-film cosines into a flat, *transposed*
  `itemSim` table (10.0 M pair updates, against 1.33 B for a naive film × film × user
  triple loop), then scores each unseen film as the rating-weighted average of the
  similarities coming from that user's 10 seed-nearest rated films.

Missing entries are **skipped, never imputed** — on a 1–5 scale a `0` can only mean
"unrated".

### Configuration

| Constant | Value | Role |
|---|---|---|
| `TOP_K` | 5 | films per list |
| `N_NEIGHBOURS` | 20 | neighbours kept for User-Based |
| `MIN_CO_RATED` | 5 | evidence floor, applied on **both** axes |
| `SEED_NEIGHBOURS` | 10 | rated films that may feed Item-Based |

## What the measurements show

Every figure below is produced by [`experiments/run.py`](experiments/run.py), which
drives the shipped page in headless Chrome and calls the real recommenders — there is
no reimplementation of the CF code behind these numbers. Section 4 of the
[report](https://github.com/PollyIva/Collaborative-Filtering-Movie-Recommender/blob/master/Polina_Ivanilova_a03_report.pdf)
has the full write-up and the ablation.

| | User-Based | Item-Based |
|---|---|---|
| Pairs to precompute | 444,153 (943²/2) | 1,413,721 (1,682²/2) |
| Pairs with no overlap | 3.4% | 30.5% |
| Pairs under 5 co-ratings | 26.2% | 70.3% |
| Query time (median, 40 users) | 5.15 ms | 1.00 ms |
| NDCG@10, held-out split (all 943 users) | **0.0383** | 0.0013 |
| Relative to random (0.0100) | **3.8× better** | 7.5× worse |

Three things fall out of that table:

- **Sparsity decides which axis you can afford.** The matrix is 93.71% empty —
  100,000 ratings spread over 1,588,752 cells. Items outnumber users 1.78×, so the
  item-item table is 3.18× larger to build, but that cost is paid **once**, whereas a
  User-Based query re-scores 20 neighbours for every candidate on every click.
- **The two lists barely agree.** Over all 943 users the two Top-5 lists share
  **11 titles in total**, and **98.8%** of users get two completely disjoint lists.
  Chance overlap would be 0.0149 titles per user; the measured figure is 0.0117.
- **Only User-Based beats chance.** Item-Based lands 7.5× *below* a random list, and
  removing its evidence floor entirely does not help (0.0009) — so the guard was never
  the cause. Cosine between 1–5 ratings is near 1.0 for almost any pair sharing five
  films, so the item-item table barely separates a Documentary from a Comedy and ends up
  ordering candidates by rater count instead of by taste. Mean-centering (Pearson) is the
  fix; report §5 has the ablation.

## Data

`u.item` and `u.data` come from the
[MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/): 943 users,
1,682 films, 100,000 ratings on a 1–5 scale. The files are stored in **Latin-1**, so
the parser decodes them with `TextDecoder('latin1')` and accented titles survive.

## Files

| File | Role |
|---|---|
| `index.html` | structure — dropdowns and the two result panels |
| `style.css` | layout and styling |
| `data.js` | fetch, parse, build the matrix and inverted indexes |
| `script.js` | cosine similarity, both recommenders, UI rendering |
| `experiments/` | headless-Chrome harness behind the measurement table |

## License

Educational project. MovieLens 100K is provided by GroupLens for research and
educational use — see the [MovieLens terms](https://grouplens.org/datasets/movielens/).
