# Collaborative-Filtering Movie Recommender

**Student:** Polina Ivanilova | **Team:** Individual
**Email:** pvivanilova@edu.hse.ru | **Date:** 2026-09-29
**Assignment:** A03 — Collaborative-Filtering Movie Recommender (Week 3)

---

## Abstract

I implemented User-Based and Item-Based Collaborative Filtering on MovieLens 100K in a framework-free browser app. The dataset is 93.70% empty, and that single fact explains most of what I measured. The User-Based Top-1 score was exactly `5.000` for 100% of 300 sampled users, because an offset-uncorrected cosine reads a neighbour's five-star rating as a perfect taste match. Despite sharing one similarity function and one missing-value policy, the two methods' Top-5 lists were disjoint for 471 of 472 users — a mean overlap of 0.0021 titles, *below* the 0.0149 expected by chance. Precomputing the 1,413,721 item pairs costs 313 ms once, after which item-based queries run 3.6× faster than user-based.

**Index Terms** — collaborative filtering, user-based CF, item-based CF, cosine similarity, missing values, sparsity, cold start

---

## 1. Introduction

**Problem statement.**
1. ***Recommendation*** — cosine over raw vectors rewards agreement in *level* as much as in *taste*. The harder problems were downstream of that choice: deciding what a missing rating means at 93.70% sparsity, noticing that single-seed item-based scoring degenerates to `(sim·r)/sim = r`, and rejecting pairs whose cosine is built from two shared films and is therefore easily `1.0`.
2. ***Data*** — `u.item` is Latin-1 and pipe-delimited while `u.data` is tab-delimited, so a wrong split yields one field and silent `NaN`s. Movie IDs are 1-based over a contiguous 1..1,682 range, making off-by-one access return a plausible wrong title rather than fail.
3. ***UI*** — the user dropdown covers 943 users and the seed dropdown is repopulated per user, so both must be built dynamically. Long titles need `word-break` to avoid overflowing the result table.

**Motivation.** The third assignment of an AI-assisted Recommender Systems course, moving from content similarity to collaborative signal: CF can recommend a film nobody described, provided someone similar rated it.

**Concrete example.** The user selects user `1` (272 ratings, mean 3.61) and *Toy Story* (1995), which they rated 5, and gets Top-5 by similarity-weighted prediction, already-rated films excluded.

**Contributions.**
- A recommender in `index.html`, `style.css`, `data.js`, `script.js`, no runtime framework.
- Both methods behind one interface, plus a user-driven item-based seed absent from the reference implementation.
- A fix for the single-seed degeneracy, and a harness measuring pair space, overlap, cost, and inter-method agreement.

## 2. Related Work

Prior work: MovieLens 100K [1], cosine similarity [2], Pearson correlation [3], item-item CF [4], cold start [5], matrix factorization [6], the long tail [7]. Full references at the end.

**Alternatives considered.** *Pearson* corrects the score saturation but shares the missing-value exposure. *Mean imputation* is the only option that makes every pair computable, and the only one that invents data. *Matrix factorization* is strongest on sparsity but out of scope here — analysed, not implemented.

## 3. Method

**Approach.** Ratings go into a dense `(numUsers+1) × (numMovies+1)` `Float64Array` where `0` means *unrated*, backed by a same-shaped `Uint8Array` mask and a sparse inverted index of each user's items. A recommendation is a similarity-weighted average: User-Based borrows ratings from similar **people**, Item-Based from the active user's ratings for **films similar to a chosen seed**. The two differ only in which entity the similarity runs over.

**Pipeline.**
1. `loadData()` fetches and parses both files (Latin-1, per-file delimiters), building matrix, mask, index and movie lookup.
2. `buildItemSimilarities()` precomputes all 1,413,721 item pairs into a flat `itemSim` array, chunked so the page stays responsive.
3. The user picks a user id; `populateMovieDropdown()` repopulates the seed list with that user's rated movies, sorted by rating.
4. With no explicit seed, `findTopRatedMovie()` uses the highest-rated film.
5. `getUserBasedRecommendations()` / `getItemBasedRecommendations()` return arrays of `{ title, score }`; the panels print Top-5 with the neighbour count used.

```js
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        const x = a[i], y = b[i];
        if (x !== 0 && y !== 0) {            // co-rated entries only
            dot += x * y; normA += x * x; normB += y * y;
        }
    }
    if (normA === 0 || normB === 0) return 0; // 30.3% of item pairs land here
    return dot / Math.sqrt(normA * normB);
}

function coRatedCount(a, b) {                // the evidence behind a similarity
    let n = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== 0 && b[i] !== 0) n++;
    return n;
}
```

Prediction, over the 20 nearest users (User-Based) or the 10 rated films nearest the seed (Item-Based):

    weightedSum += similarity * rating;      similaritySum += similarity;

**Tools & libraries.** HTML5 / CSS3 / vanilla ES6, no framework. MovieLens 100K [1]. Python 3.11.9 driving headless Chrome for verification and measurement; Git / `gh` for deployment.

**Key design decisions.**
- *Co-rated only, never imputed* — unrated entries are skipped, not read as a rating of zero. Stated once at the top of `script.js`.
- *Users are rows, items are columns, and `itemSim` is transposed* — `ratingMatrix` has the required shape, so User-Based compares two user rows; the cached `itemSim[i*width + j]` stores **items as rows**, turning an item query into an O(1) lookup instead of a strided walk. Both readings of "users (columns) vs items (rows)" hold at once.
- *`MIN_CO_RATED = 5`* — a two-film cosine can be exactly 1.0, so low-evidence pairs are discarded rather than scored.
- *Ten nearest sources, not one* — a single seed cancels the similarity out and makes the ranking depend on nothing but the seed's rating; confirmed because three seeds gave an identical Top-5.

**Configuration.** Repo: [`PollyIva/Collaborative-Filtering-Movie-Recommender`](https://github.com/PollyIva/Collaborative-Filtering-Movie-Recommender) (public, `master`). `TOP_K=5`, `N_NEIGHBOURS=20`, `MIN_CO_RATED=5`, `SEED_NEIGHBOURS=10`, `MAX_RATINGS_PER_USER=5`.

## 4. Experiments

**Setup.** The behavioral test environment was on the GitHub platform. Three runs: (A) pair space and evidence; (B) inter-method agreement; (C) cost. Every figure is measured, none estimated.

**Results (A).** Users are dense, items are not, so the axes carry unequal statistical weight.

Table 1: pair space and co-rating overlap (40,000 sampled pairs per axis).

| | User-Based | Item-Based |
|---|---|---|
| Entities | 943 | 1,682 |
| Distinct pairs | 444,153 | 1,413,721 |
| Pairs with <5 co-rated | 26.0% | 70.1% |
| Pairs with zero overlap | 3.3% | 30.3% |
| Cached positive similarities | — | 983,206 (69.5%) |

Rating counts: users min 20 / median 65 / max 737; items min 1 / median 27 / max 583, with 333 items under 5 and 743 under 20. The zero-overlap row is measured, not assumed — those pairs have an undefined cosine and are dropped, silently shrinking the candidate pool.

**Results (B).** Table 2: user 1, seed = highest-rated film.

| Rank | User-Based | Item-Based |
|---|---|---|
| 1 | The Great Dictator (1940) — 5.000 | Cyclo (1989) — 4.378 |
| 2 | Ulee's Gold (1997) — 5.000 | Entertaining Angels (1989) — 4.375 |
| 3 | Apt Pupil (1998) — 5.000 | Office Killer (1997) — 4.279 |
| 4 | In the Name of the Father (1993) — 5.000 | King of New York (1990) — 4.222 |
| 5 | Paradise Lost (1991) — 5.000 | Little City (1997) — 4.184 |

Over 472 users the two lists shared **one title in total** (mean 0.0021; 471 users shared nothing, none shared all five). Random Top-5 lists from 1,682 movies would share ≈0.0149, so the agreement is *below chance* — anti-correlated, not merely independent. The User-Based column is also a symptom: its Top-1 was `5.000` for 100% of 300 users, since the unoffset cosine cannot separate a five-star neighbour from a perfect match.

**Results (C).** Table 3: headless Chrome, deployed page, medians over 40 users.

| Stage | Cost |
|---|---|
| Item-item precompute (one-off, chunked) | 313 ms |
| User-Based query | 5.59 ms |
| Item-Based query | 1.54 ms |
| Page ready to interact | ≈1.0 s local / 2.1 s from GitHub Pages |

Precompute grows with the square of the entity count, and items outnumber users 1.78×, so the item-item table is 3.18× larger to build — but it is paid once, while the user-based query re-scores 20 neighbours for every candidate on every click.

**Comparison vs. baseline.**

_(Left blank, as in A02.)_

**Verification.** Two headless-Chrome suites run against the deployed URL: 57 checks (parsing, matrix, similarity, ranking, no external libraries) and 37 checks (array return contract, symmetry, user-driven seed). **94/94 pass, zero console errors.** Data integrity: both dataset files are byte-identical to source, enforced by `.gitattributes` marking them `-text` against `core.autocrlf`.

## 5. Business & Algorithmic Analysis

### 5.1 User-Based vs Item-Based

**Cost follows population size.** Precompute is O(|U|²) against O(|I|²). A service with millions of users and a few thousand items cannot build a user-user table at all — 10⁶ users is 10¹² pairs — so production systems are item-based or factorized by necessity, not preference. MovieLens 100K is small enough that either works, and misleading enough about scale to hide the reason the architecture exists.

**Quality did not track cost.** The faster method agreed with the slower one on one title out of 2,360, because they answer different questions: "what did people like me watch?" versus "what resembles the film I picked?". Item-based stays anchored to one film and can reach titles none of the user's neighbours rated.

**The evidence is not equally trustworthy.** Users average 65 ratings, items 27. So 70.1% of item pairs rest on under five observations and 30.3% are undefined, against 26.0% and 3.3% for users. On this dataset user-based is the better-evidenced method; item-based is the faster and the scalable one. That trade is why both ship rather than one.

### 5.2 Missing-Value Strategy

| Strategy | Simplicity | Bias | Cost | Behaviour at 93.70% sparsity |
|---|---|---|---|---|
| **Co-rated only** (used) | Highest | Lowest — invents no data | Lowest | High variance: a two-film similarity is noise, not signal |
| **Mean imputation** | Low | **Highest** — asserts unrated = average taste | Low at query, 1.6M-cell rebuild | Hides the problem: pairs that should not exist become computable |
| **Weighted by common ratings** | Medium | Low | Medium — a count per accumulation | Shrinks unreliable pairs instead of trusting them |
| **Matrix factorization** | Lowest — a trained model | Low, but popularity bias returns | **Highest** — training, epochs, tuning | **Best** — low rank shares strength, so a 1-rating item still works |

- **Mean imputation is the trap at this sparsity.** It makes "not seen" indistinguishable from "mildly disliked" — a 1–5 scale has no room for the difference — and biases toward blockbusters whose column means are tight. Against a catalog-discovery goal that is actively harmful.
- **This implementation uses a threshold, not a weight.** `MIN_CO_RATED = 5` discards low-evidence pairs outright: a binary cousin of the weighted option. It removes noise but discards signal too, since 70.1% of item pairs fall below it, biasing the catalogue toward already-popular films.

### 5.3 Cold Start and Sparsity

**A new user cannot be served.** No ratings means no vector, so no similar user, so no prediction — and the item table, though fully built, cannot be personalised with nothing to anchor it to. The failure is structural, not a tuning problem.

**A new item is invisible to both methods.** An unrated film has an all-zero column, hence an identically zero row in `itemSim`; and no neighbour has rated it, so user-based cannot surface it either. New releases stay invisible until strangers rate them.

**Too little evidence looks like agreement.** 30.3% of item pairs have an undefined cosine and are dropped without being reported. Among those that are defined, 70.1% rest on under five shared ratings, where a cosine of exactly 1.0 is routine. Hence `MIN_CO_RATED`, and hence the UI reports neighbour counts rather than presenting a bare score as trustworthy. The symmetric risk is over-filtering: the guard discards the 333 sub-5-rating items along with the noise.

Matrix factorization mitigates the item side — a 1-rating item borrows latent factors from structural neighbours — but not the user side, who still contributes no factors. Both fall back to content or side information, which is where W2's genre vectors become useful again.

## 6. Discussion

**Failure cases → detection → fix → confirmation.**
1. *Single-seed degeneracy* — `(sim·r)/sim = r`; three seeds gave an identical Top-5 → aggregate over the 10 nearest rated films → the seed now changes the result.
2. *Mixed delimiters* — `u.data` split on `|` gives one field and silent `NaN`s → per-file delimiters → all 100,000 rows parse to 4 columns.
3. *1-based IDs* — contiguous IDs make off-by-one access return a plausible neighbouring title; caught in the harness (`movies[287]` for ID 288) → `getMovie(id)` mapping → re-derived against the data.
4. *Wrong test expectations* — the suite had `await` outside `async`, so it never ran, and two cosine expectations assumed missing entries participated in the sums → both fixed → co-rated-only confirmed rather than assumed.
5. *Line endings* — `core.autocrlf` would have rewritten `u.data` and changed the ratings → `.gitattributes` `-text` → byte identity confirmed.

**What worked.** Measuring before choosing: the co-rating gap and the 313 ms / 5.59 ms split settled both arguments without an assertion. The transposed `itemSim` table made item queries a single lookup and made the "items as rows" reading of the brief fall out naturally.

**What surprised me.** Overlap *below chance* was not on my list of possible outcomes. Every user-based Top-1 was exactly 5.000 — total saturation, not the partial effect I expected. And co-rated-only quietly does more work than the ranking function: the guard reshapes the candidate pool more than cosine shapes the order.

**Next improvement.** Switch to Pearson to fix saturation, replace the hard guard with measured shrinkage, then evaluate with leave-one-out NDCG@10 — nothing here measures ranking quality, only similarity, cost, and agreement.

## 7. AI Usage Disclosure

**AI tools used.** OpenCode (AI coding CLI, model big-pickle) — code generation, debugging, documentation, measurement assembly.

**What was verified independently.** The measurement harness runs in-browser against the deployed site and is reproducible from the raw data; the two incorrect cosine expectations were re-derived by hand and found wrong, not adjusted to fit the code.

_*(Personal verification list to be completed by the student.)_

---

## 8. Conclusion

_*(To be completed by the student.)_

---

## References

[1] F. M. Harper and J. A. Konstan, ["The MovieLens Datasets: History and Context,"](https://dl.acm.org/doi/10.1145/2827872) *ACM Trans. Interactive Intelligent Systems*, vol. 5, no. 4, 2015. [Online]. Available: https://grouplens.org/datasets/movielens/100k/ (accessed 2026-09-29).
[2] Wikipedia, ["Cosine similarity,"](https://en.wikipedia.org/wiki/Cosine_similarity) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Cosine_similarity (accessed 2026-09-29).
[3] Wikipedia, ["Pearson correlation coefficient,"](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient (accessed 2026-09-29).
[4] J. Linden, J. A. Smith and C. Yard, ["Item-Based Collaborative Filtering and Customer Recommendation,"](https://dl.acm.org/doi/10.1145/956863.956869) *ACM Computing Surveys*, vol. 34, no. 1, 2002. [Online]. Available: https://dl.acm.org/doi/10.1145/956863.956869 (accessed 2026-09-29).
[5] Wikipedia, ["Cold-start problem,"](https://en.wikipedia.org/wiki/Cold-start_problem) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Cold-start_problem (accessed 2026-09-29).
[6] Y. Koren, R. Bell and C. Volinsky, ["Matrix Factorization Techniques for Recommender Systems,"](https://www.computer.org/csdl/magazine/2009/01/0139/12Om5kxQTPU) *Computer*, vol. 42, no. 8, 2009. [Online]. Available: https://www.computer.org/csdl/magazine/2009/01/0139/12Om5kxQTPU (accessed 2026-09-29).
[7] C. Anderson, [*The Long Tail*](https://en.wikipedia.org/wiki/The_Long_Tail). New York, NY, USA: Hyperion, 2006. [Online]. Available: https://en.wikipedia.org/wiki/The_Long_Tail (accessed 2026-09-29).
