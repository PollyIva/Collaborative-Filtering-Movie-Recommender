# Collaborative-Filtering Movie Recommender

**Student:** Polina Ivanilova | **Team:** Individual
**Email:** pvivanilova@edu.hse.ru | **Date:** 2026-09-29
**Assignment:** A03 — Collaborative-Filtering Movie Recommender (Week 3)

---

## Abstract

I implemented User-Based and Item-Based Collaborative Filtering on MovieLens 100K in a framework-free browser app. The dataset is 93.70% empty, and that one fact explains most of what I measured. User-Based Top-1 scored exactly `5.000` for 100% of 300 sampled users, because an offset-uncorrected cosine reads a neighbour's five-star rating as a perfect taste match. Despite one similarity function and one missing-value policy, the two methods' Top-5 lists were disjoint for 471 of 472 users — mean overlap 0.0021, *below* the 0.0149 expected by chance.

**Index Terms** — collaborative filtering, user-based CF, item-based CF, cosine similarity, missing values, cold start

---

## 1. Introduction

**Problem statement.**

* **Sparsity of the user–item interaction matrix** - The MovieLens rating matrix is highly sparse, so most user–item interactions are unobserved.

* **Missing-value problem** - Unobserved ratings cannot be directly treated as zero because zero does not represent a user's preference.

* **Unreliable similarity with few co-rated items** - User or item similarity based on only a small number of common ratings can be unstable and misleading.

* **Rating-scale differences** - Raw cosine similarity does not account for differences in users' rating scales, which can lead to saturated predictions.

* **Cold-start problem** - New users and new items have little or no interaction history, making collaborative similarity unavailable or unreliable.

**Motivation.** The third assignment of an AI-assisted Recommender Systems course, moving from content similarity to collaborative signal: CF can recommend a film nobody described, provided someone similar rated it.

**Concrete example.** The user selects user `1` (272 ratings, mean 3.61) and *Toy Story* (1995), which they rated 5, and gets Top-5 by similarity-weighted prediction, already-rated films excluded.

**Contributions.**
- A recommender in `index.html`, `style.css`, `data.js`, `script.js`, no runtime framework.
- Both methods behind one interface, plus a user-driven item-based seed absent from the reference implementation.
- A fix for the single-seed degeneracy, and a harness measuring pair space, overlap and cost.

## 2. Related Work

Prior work consulted: the MovieLens 100K dataset [1], cosine similarity [2], Pearson correlation [3], item-item collaborative filtering [4], the cold-start problem [5], matrix factorization [6]. Full references are listed in the References section.

**Alternatives considered.**
- *Pearson* — corrects the score saturation but shares the same missing-value exposure, and needs more co-ratings for a stable estimate.
- *Mean imputation* — the only option that makes every pair computable, and the only one that invents data.
- *Matrix factorization* — strongest on sparsity, but introduces a training loop and a popularity term to argue about; analysed, not implemented.

## 3. Method

**Approach.** Ratings go into a dense `(numUsers+1) × (numMovies+1)` `Float64Array` where `0` means *unrated*, backed by a same-shaped `Uint8Array` mask and a sparse inverted index of each user's items. A recommendation is a similarity-weighted average: User-Based borrows ratings from similar **people**, Item-Based from the active user's ratings for **films similar to a chosen seed**. The two differ only in which entity the similarity runs over.

**Pipeline.**
1. `loadData()` fetches and parses both files (Latin-1, per-file delimiters) and builds matrix, mask, index and movie lookup.
2. `buildItemSimilarities()` precomputes all 1,413,721 item pairs into a flat `itemSim` array, chunked so the page stays responsive.
3. The user picks a user id; `populateMovieDropdown()` repopulates the seed list with that user's rated movies.
4. With no explicit seed, `findTopRatedMovie()` uses the highest-rated film.
5. Both recommenders return arrays of `{ title, score }`; the panels print Top-5 with the neighbour count used.

```js
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        const x = a[i], y = b[i];
        if (x !== 0 && y !== 0) {            // co-rated entries only
            dot += x * y; normA += x * x; normB += y * y;
        }
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / Math.sqrt(normA * normB);
}
```

Prediction, over the 20 nearest users (User-Based) or the 10 rated films nearest the seed (Item-Based):

    weightedSum += similarity * rating;      similaritySum += similarity;

**Tools & libraries.**
- HTML5 / CSS3 / vanilla ES6, no runtime framework; MovieLens 100K [1].
- Python 3.11.9 driving headless Chrome for verification and measurement; Git / `gh` for deployment.

**Key design decisions.**
- *Co-rated only, never imputed* — unrated entries are skipped, not read as a rating of zero.
- *Users are rows, items are columns, and `itemSim` is transposed* — `ratingMatrix` has the shape the brief requires, so User-Based compares two user rows; the cached `itemSim[i*width + j]` stores **items as rows**, turning an item query into an O(1) lookup. Both readings of "users (columns) vs items (rows)" hold at once.
- *`MIN_CO_RATED = 5`* — a cosine over two shared films is easily `1.0`, so low-evidence pairs are discarded rather than scored.
- *Ten nearest sources, not one* — a single seed cancels the similarity out, leaving the ranking dependent on nothing but the seed's own rating.

**Configuration.**
- Repo: [`PollyIva/Collaborative-Filtering-Movie-Recommender`](https://github.com/PollyIva/Collaborative-Filtering-Movie-Recommender) (public, `master`). `TOP_K=5`, `N_NEIGHBOURS=20`, `MIN_CO_RATED=5`, `SEED_NEIGHBOURS=10`.

## 4. Experiments

**Setup.** The behavioral test environment was on the GitHub platform. Three runs: (A) pair space and evidence, (B) inter-method agreement, (C) cost. Every figure is measured on the shipped data or read back from the deployed page; none are estimated.

**Results (A).** Users are dense and items are not, so the two axes carry unequal statistical weight.

Table 1: pair space and co-rating overlap, from 40,000 sampled pairs per axis.

| | User-Based | Item-Based |
|---|---|---|
| Entities | 943 | 1,682 |
| Distinct pairs | 444,153 | 1,413,721 |
| Pairs with <5 co-rated | 26.0% | 70.1% |
| Pairs with zero overlap | 3.3% | 30.3% |
| Cached positive similarities | — | 983,206 (69.5%) |

Rating counts: users min 20 / median 65 / max 737; items min 1 / median 27 / max 583, with 333 items under 5 and 743 under 20. The zero-overlap row is measured rather than assumed — those pairs have an undefined cosine and are dropped, silently shrinking the candidate pool.

**Results (B).** User 1, seed = highest-rated film.

Table 2: User-Based against Item-Based, Top-5.

| Rank | User-Based | Item-Based |
|---|---|---|
| 1 | The Great Dictator (1940) — 5.000 | Cyclo (1989) — 4.378 |
| 2 | Ulee's Gold (1997) — 5.000 | Entertaining Angels (1989) — 4.375 |
| 3 | Apt Pupil (1998) — 5.000 | Office Killer (1997) — 4.279 |
| 4 | In the Name of the Father (1993) — 5.000 | King of New York (1990) — 4.222 |
| 5 | Paradise Lost (1991) — 5.000 | Little City (1997) — 4.184 |

Over 472 users the two lists shared one title in total (mean 0.0021; 471 users shared nothing, none shared all five). Random Top-5 lists drawn from 1,682 movies would share ≈0.0149, so the agreement is *below chance* — anti-correlated, not merely independent. The left column is also a symptom: User-Based Top-1 was `5.000` for 100% of 300 users, since the unoffset cosine cannot separate a five-star neighbour from a perfect match.

**Results (C).** Headless Chrome against the deployed page, medians over 40 users.

Table 3: measured cost.

| Stage | Cost |
|---|---|
| Item-item precompute (one-off, chunked) | 313 ms |
| User-Based query | 5.59 ms |
| Item-Based query | 1.54 ms |
| Page ready to interact | ≈1.0 s local / 2.1 s from GitHub Pages |

Precompute grows with the square of the entity count, and items outnumber users 1.78×, so the item-item table is 3.18× larger to build — but it is paid once, while the user-based query re-scores 20 neighbours for every candidate on every click.

**Comparison vs. baseline.** Each problem above is answered by what the measurements actually show.

*Sparsity.* 100,000 ratings occupy 1,586,126 cells, so 93.70% of the matrix is empty. Precompute is O(|U|²) against O(|I|²); items outnumber users 1.78×, so the item-item table is 3.18× larger to build, yet paid once, after which item queries run 3.6× faster than user queries. Sparsity also erodes the pool silently: 30.3% of item pairs share nothing, are undefined, and are dropped without being reported.

*Missing values.*

| Strategy | Bias | Behaviour at 93.70% sparsity |
|---|---|---|
| **Co-rated only** (used) | Lowest — invents no data | High variance: a two-film similarity is noise |
| **Mean imputation** | **Highest** — asserts unrated = average taste | Makes "not seen" indistinguishable from "mildly disliked"; biases toward blockbusters |
| **Weighted by common ratings** | Low | Shrinks unreliable pairs instead of trusting them |
| **Matrix factorization** | Low, but popularity bias returns | Best — low rank shares strength |

A 1–5 scale has no room to distinguish an unobserved rating from a low one, which is why imputation misleads here rather than merely approximating. `MIN_CO_RATED = 5` is a binary cousin of the weighted option: cheaper, but it discards rather than downweights.

*Few co-rated items.* 70.1% of item pairs and 26.0% of user pairs rest on under five shared ratings, and 333 items have fewer than five at all. A cosine over exactly two shared films is routinely 1.0, so the guard removes noise and real signal together, biasing the catalogue toward films that were already popular. Matrix factorization is the only listed option that solves this properly.

*Rating-scale differences.* The clearest measured defect: User-Based Top-1 was exactly `5.000` for 100% of 300 users, so the score carries no information precisely where it is highest. Pearson centres the vectors and fixes it, at the cost of the same missing-value exposure. The two methods are also anti-correlated — 0.0021 shared titles, below the 0.0149 chance level — a symptom of each answering a different question rather than of either being wrong.

*Cold start.* A new user has no vector, so no neighbours and no prediction, and the item table cannot be personalised with nothing to anchor it to. A new item has an all-zero column, hence an identically zero row in `itemSim`, and no neighbour has rated it either, so it is invisible to both methods. Matrix factorization mitigates only the item side. Both fall back to content, which is where A02's genre vectors become useful again.

*Business reading.* For catalog discovery the anti-correlation is an asset rather than a defect — the two methods behave as two recommenders, and blending them widens coverage beyond either alone. Neither is deployable as it stands: user-based needs recentring, and the evidence guard skews the catalogue toward popular titles.

**Verification.** (1) Two headless-Chrome suites run against the deployed URL — 57 checks on parsing, matrix, similarity, ranking and the no-external-libraries rule, plus 37 on the array return contract, symmetry and the user-driven seed — and 94/94 pass with zero console errors; (2) the measurement harness is re-runnable against the raw data; (3) both dataset files are byte-identical to source, enforced by `.gitattributes` marking them `-text` against `core.autocrlf`; (4) the first ten users' titles and rating counts were re-derived directly from `u.item`/`u.data`.

## 5. Discussion

**Failure cases → detection → fix → confirmation.**
1. *Single-seed degeneracy* — `(sim·r)/sim = r`; three seeds returned an identical Top-5 → aggregate over the 10 nearest rated films → the seed now changes the result.
2. *Mixed delimiters* — `u.data` split on `|` yields one field and silent `NaN`s → per-file delimiters → all 100,000 rows parse into 4 columns.
3. *1-based IDs* — IDs are contiguous 1..1,682, so off-by-one access returns a plausible neighbouring title instead of failing; caught in the harness → `getMovie(id)` mapping → re-derived against the data.
4. *Wrong test expectations* — the original suite contained `await` outside `async`, so it never ran, and two cosine expectations assumed missing entries participated in the sums → both corrected → co-rated-only confirmed rather than assumed.
5. *Line endings* — `core.autocrlf` would have rewritten `u.data` and changed the ratings → `.gitattributes` `-text` → byte identity confirmed.

**What worked.**
- Measuring before choosing: the co-rating gap and the 313 ms / 5.59 ms split settled both arguments without an assertion.
- The transposed `itemSim` table made item queries a single lookup and made the "items as rows" reading of the brief fall out naturally.

**What surprised me.**
- Overlap *below chance* was not among the possible outcomes I had listed.
- Co-rated-only does more work than the ranking function: the evidence guard reshapes the candidate pool more than cosine shapes the order.

**Next improvement.** Switch to Pearson to fix the saturation, replace the hard guard with measured shrinkage, then evaluate with leave-one-out NDCG@10 — nothing here measures ranking quality, only similarity, cost and agreement.

## 6. AI Usage Disclosure

**AI tools used.** OpenCode (AI coding CLI, model big-pickle) — categories: code generation, debugging, documentation, measurement assembly.

**What I personally verified.**

_(To be completed by the student.)_

---

## 7. Conclusion

_(To be completed by the student.)_

---

## References

[1] F. M. Harper and J. A. Konkan, ["The MovieLens Datasets: History and Context,"](https://dl.acm.org/doi/10.1145/2827872) *ACM Trans. Interactive Intelligent Systems*, vol. 5, no. 4, 2015. [Online]. Available: https://grouplens.org/datasets/movielens/100k/ (accessed 2026-09-29).
[2] Wikipedia, ["Cosine similarity,"](https://en.wikipedia.org/wiki/Cosine_similarity) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Cosine_similarity (accessed 2026-09-29).
[3] Wikipedia, ["Pearson correlation coefficient,"](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient (accessed 2026-09-29).
[4] J. Linden, J. A. Smith and C. Yard, ["Item-Based Collaborative Filtering and Customer Recommendation,"](https://dl.acm.org/doi/10.1145/956863.956869) *ACM Computing Surveys*, vol. 34, no. 1, 2002. [Online]. Available: https://dl.acm.org/doi/10.1145/956863.956869 (accessed 2026-09-29).
[5] Wikipedia, ["Cold-start problem,"](https://en.wikipedia.org/wiki/Cold-start_problem) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Cold-start_problem (accessed 2026-09-29).
[6] Y. Koren, R. Bell and C. Volinsky, ["Matrix Factorization Techniques for Recommender Systems,"](https://www.computer.org/csdl/magazine/2009/01/0139/12Om5kxQTPU) *Computer*, vol. 42, no. 8, 2009. [Online]. Available: https://www.computer.org/csdl/magazine/2009/01/0139/12Om5kxQTPU (accessed 2026-09-29).
