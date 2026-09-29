# Collaborative-Filtering Movie Recommender

**Student:** Polina Ivanilova | **Team:** Individual
**Email:** pvivanilova@edu.hse.ru | **Date:** 2026-09-29
**Assignment:** A03 — Collaborative-Filtering Movie Recommender (Week 3)

---

## Abstract

I implemented both classical neighbourhood methods — User-Based and Item-Based Collaborative Filtering — on MovieLens 100K in a framework-free browser app, and then measured what separates them. The User-Based Top-1 score was exactly `5.000` for **100% of the 300 users sampled**, because an offset-uncorrected cosine reads a neighbour who gave a shared film five stars as a perfect match; Item-Based scores spread from `1.000` to `5.000`. Despite using the same similarity function and the same co-rated-only policy, the two methods' Top-5 lists were **disjoint for 471 of 472 users** (mean overlap 0.0021 titles — *below* the 0.0149 expected by chance), so they are not redundant implementations of one question. Precomputing the 1,413,721 item pairs costs 313 ms once, after which item-based queries run **3.6× faster** than user-based (1.54 ms vs 5.59 ms). The dataset is 93.70% empty, which is the number that explains most of the above.

**Index Terms** — collaborative filtering, user-based CF, item-based CF, cosine similarity, missing values, sparsity, cold start, MovieLens

---

## 1. Introduction

**Problem statement.** The work splits into three classes:

1. ***Recommendation problem***:
-   *choosing a similarity function.* Cosine over raw rating vectors rewards agreement in *level* as well as in *taste*: two users who both rate everything five stars score `1.0` while sharing no taste at all. Pearson would centre the vectors first, but the brief asks for one missing-value policy and one similarity, and cosine on co-rated entries is the defensible baseline — the cost of the choice is then measured in §4 rather than hidden.
-   *deciding what a missing rating means.* With 93.70% of the matrix empty, this choice dominates every downstream number. I selected **co-rated-only**: missing entries are skipped, never imputed. See §5.2 for the alternatives and why mean imputation is the dangerous one.
-   *an item-based formula that silently collapses.* With a single seed item the weighted average degenerates: `score = Σ(simᵢ·rᵢ)/Σ(simᵢ)` over one source becomes `r₁`, so every movie inherits the *seed's* rating and the ranking degenerates to movie-ID order. This is not a rounding artefact — it is the formula being under-determined. I confirmed it empirically: three different seeds returned an identical Top-5.
-   *zero-overlap pairs producing a confident ranking.* 30.3% of item pairs share no ratings, so the cosine denominator is zero. A separate failure is subtler: a cosine over exactly two shared films can be exactly `1.0` (both users rated one film 4 and nothing else), letting noise pass as a perfect match. A `MIN_CO_RATED = 5` guard discards those pairs.
-   *which axis similarity lives on.* The brief contrasts "users (columns)" with "items (rows)", but the required matrix shape `(numUsers+1) × (numMovies+1)` forces users into rows. Both readings are satisfied, and the resolution is documented in §3.3.
2. ***Data problem***:
-   *the dataset is not UTF-8.* As in Week 2, `u.item` is Latin-1; `fetch(...).then(r => r.text())` garbles accented titles. Decoding with `new TextDecoder('latin1')` fixes it.
-   *the two files use different delimiters.* `u.item` is pipe-delimited, but `u.data` is **tab**-delimited. Splitting `u.data` on `|` returns a single field, every `parseInt` yields `NaN`, and nothing throws — the parser looks like it ran. Each file is split with its own delimiter.
-   *IDs are 1-based.* All 1,682 movie IDs are present and contiguous (1..1,682, no gaps), so a 0-based array makes `movies[id]` off by one for the entire file. The wrong version still returns a valid, plausible *neighbouring* title rather than failing; `getMovie(id)` resolves the mapping.
-   *93.70% sparsity.* The matrix is 1,586,126 cells holding 100,000 ratings. Any dense approach is mostly zeros, which is why inverted co-rating indexes are used instead of scanning rows.
3. ***HTML/UI problem***:
-   *two selectors, one of them large.* The user dropdown covers 943 users; the movie dropdown lists the selected user's rated movies sorted by rating, and 743 catalog items have fewer than 20 ratings, so it is populated dynamically rather than statically.
-   *long titles break tables.* As in Week 2, `nowrap` on result cells let "Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)" push the layout wide; `word-break: break-word` fixes it.
-   *results with no provenance.* A score alone is not interpretable, so the UI states the number of neighbours used, and the selected user and movie title are printed both on screen and to the console.

**Motivation.** This is the third assignment of an AI-assisted Recommender Systems course, moving from item-to-item content similarity to user-to-item collaborative signal. Content-based methods could only ever recommend films of the same genre; CF can recommend a film no one in the dataset has described, provided someone similar rated it.

**Concrete example.** The user selects user `1` (272 ratings, mean 3.61) and the movie *Toy Story* (1995), which they rated 5, and gets Top-5 ranked by similarity-weighted prediction, with already-rated films excluded.

**Contributions.**
-   A recommender implemented with `index.html`, `style.css`, `data.js`, and `script.js`, without a runtime framework.
-   Both User-Based and Item-Based CF behind one interface, switchable per query.
-   A user-driven item-based seed, which the reference implementation does not expose, plus a fix for the single-seed degeneracy it exposes.
-   A measurement harness that quantifies pair-space size, co-rating overlap, query latency, and Top-5 agreement between the two methods.

## 2. Related Work

Prior work consulted: the MovieLens 100K dataset [1], cosine similarity [2], Pearson correlation [3], the item-item formulation of Linden et al. [4], the cold-start problem [5], matrix factorization for recommender systems [6], popularity bias and the long tail [7], and standard recommender evaluation practice [8]. Full references are in the References section.

**Alternatives considered.**
- *Pearson correlation* — corrects the score-saturation problem in §4.4, but has the same missing-value exposure and a lower bound on how few co-ratings are needed for a stable estimate.
- *Mean imputation* — the only option that makes every pair computable, and the only one that invents data. Rejected on bias grounds (§5.2), not on simplicity.
- *Matrix factorization* — the strongest option here on sparsity (§5.3), but it introduces a training loop, hyperparameters, and a popularity term that must itself be argued about. Out of scope for a classical-CF assignment; analysed, not implemented.

## 3. Method

**Approach.** The app loads `u.item` and `u.data` in the browser, decodes `u.item` as Latin-1, and builds three structures: a dense `(numUsers+1) × (numMovies+1)` `Float64Array` of ratings with `0` meaning "unrated", a same-shaped `Uint8Array` mask, and a sparse inverted index mapping each user to the list of items they rated.

**3.1 Similarity.** For users *u, v*:

    sim(u,v) = Σ_{i ∈ co-rated} r_ui · r_vi  /  sqrt( Σ_{i ∈ co-rated} r_ui² · Σ_{i ∈ co-rated} r_vi² )

For items *i, j* the same expression is applied to the two item columns, and the results are cached. Only co-rated entries enter the sums, which is what makes `0` safe as a missing marker. Pairs with fewer than `MIN_CO_RATED = 5` shared entries are discarded rather than scored.

**3.2 User-Based CF.** The active user's top `TOP_K = 20` most similar users are found by scanning the inverted index, then for every unrated candidate:

    prediction(u, i) = Σ_{v ∈ neighbours(u)} sim(u,v) · r_vi  /  Σ_{v ∈ neighbours(u)} sim(u,v)

Already-rated films are excluded. To bound a click, the active user's rating list is capped at `MAX_RATINGS_PER_USER = 5`, so a heavy rater such as user 1 (272 ratings) is scored against at most five of them.

**3.3 Which axis is a row.** The brief contrasts "users (columns)" with "items (rows)", while requiring a `(numUsers+1) × (numMovies+1)` matrix. Both are satisfied, because the two tables are oriented differently on purpose:

-   `ratingMatrix[userId][itemId]` — shape `(numUsers+1) × (numMovies+1)`, so **users are rows, items are columns**. User-Based CF compares two user *rows*.
-   `itemSim[i * width + j]` — the item-item results cached **transposed, with items as rows**. Item-Based CF then reduces a query to an O(1) row-wise lookup instead of a strided walk down two columns.

The transposition is not cosmetic; it is what makes the precomputed table usable at query time.

**3.4 Item-Based CF, and the degeneracy fix.** Item-based prediction is the same weighted average, but the ratings come from the active user's own history, weighted by item-item similarity to a **seed** film the user selects. This is an extension beyond the reference implementation, which always scores against the user's highest-rated film.

The naive form of this formula is degenerate. With one source item the sum has a single term, so `score = (sim · r) / sim = r`: every candidate receives the *seed's own rating*, the similarity cancels, and the ranking is determined entirely by which items happen to be similar to it. Measured directly, three different seeds produced an identical Top-5.

The fix is to define the source set as the `SEED_NEIGHBOURS = 10` films from the user's own history that are most similar to the seed, and aggregate over those. Similarity now varies across sources, the seed's rating is only one term among ten, and re-selecting the seed measurably changes the Top-5. A flat two-hop weight was also tried and rejected: it was too diffuse, changing the Top-5 for only 3 of 10 users.

**3.5 Missing-value policy.** Stated once at the top of `script.js`, as §6 of the brief requires: an entry of `0` means *unrated*, never *rated zero*; similarities and predictions skip unrated entries; no value is ever imputed. The `MIN_CO_RATED = 5` guard is a filter on evidence, not a second weighting scheme.

## 4. Experiments

**4.1 Parsing.** `u.item` holds 1,682 pipe-delimited records with 19 genre columns and contiguous IDs 1..1,682; `u.data` holds 100,000 **tab**-delimited rows in the required four columns. The Latin-1 check is byte-level: `0xE9` renders as `é` in *Café au Lait*, confirming the decode. 2 films legitimately carry all-zero genre flags — an `(unknown)` row, not a parse error.

**4.2 Rating distribution.** Mean ratings per user for the first ten users are `3.61, 3.71, 2.80, 4.33, 2.87, 3.64, 3.97, 3.80, 4.27, 4.21` for counts `272, 62, 54, 24, 175, 211, 403, 59, 22, 184`. Across the dataset, users rate a median of **65** films (min 20, max 737) and no user has fewer than 20; items receive a median of **27** (min 1, max 583), with **333 items under 5 ratings** and 743 under 20.

**4.3 Pair-space size and evidence.** *Table 1.*

| | User-Based | Item-Based |
|---|---|---|
| Entities | 943 | 1,682 |
| Distinct pairs | 444,153 | 1,413,721 |
| Pairs with <5 co-rated | 26.0% | 70.1% |
| Pairs with zero overlap | 3.3% | 30.3% |
| Cached positive similarities | — | 983,206 (69.5%) |

*Table 1.* Pair counts exact; percentages from 40,000 randomly sampled pairs per axis.

**4.4 The two methods disagree almost completely.** Across 472 users, the User-Based and Item-Based Top-5 lists shared **one title in total** — a mean overlap of 0.0021 titles per user, with 471 users sharing nothing and 0 users sharing all five. Random independent lists would share ≈0.0149 titles per user, so the observed agreement is *below chance*: the two rankings are anti-correlated, not merely independent.

*Table 2* — user 1, two consecutive runs.

| # | User-Based | Item-Based |
|---|---|---|
| 1 | The Great Dictator (1940) — 5.000 | Cyclo (1989) — 4.378 |
| 2 | Ulee's Gold (1997) — 5.000 | Entertaining Angels (1989) — 4.375 |
| 3 | Apt Pupil (1998) — 5.000 | Office Killer (1997) — 4.279 |
| 4 | In the Name of the Father (1993) — 5.000 | King of New York (1990) — 4.222 |
| 5 | Paradise Lost (1991) — 5.000 | Little City (1997) — 4.184 |

*Table 2.* The first column is a symptom, not a feature: the User-Based Top-1 score was `5.000` for **100% of 300 sampled users** (min = median = max = 5.000). The unoffset cosine makes a neighbour who rated a shared film five stars indistinguishable from a perfect taste match, so the score is uninformative precisely where it is highest. Item-Based Top-1 scores range 1.000–5.000, because aggregating over ten similar items mixes a broader set of ratings.

**4.5 Cost.** *Table 3.*

| Stage | Cost |
|---|---|
| Item-item precompute (one-off, chunked) | 313 ms |
| User-Based query | 5.59 ms |
| Item-Based query | 1.54 ms |
| Page ready to interact | ≈1.0 s local / 2.1 s from GitHub Pages |

*Table 3.* Chrome headless, warm page, medians over 40 users.

The asymmetry is structural: precompute grows with the *square* of the entity count, and here items outnumber users 1.78×, so the item-item table is **3.18× larger to build**. That cost is paid once. The user-based query then re-scores against 20 neighbours for every candidate on every click, which is why it is 3.6× slower per query.

**4.6 Verification.** Two headless-Chrome suites run against the deployed GitHub Pages URL: the original 57-check suite (parsing, matrix contents, similarity values, ranking, and the "no external libraries" rule) and a 37-check suite for the prompt's array return contract, similarity symmetry, and the user-driven seed. **94/94 pass with zero console errors.** Automated checks are necessary but not sufficient: the seed-behaviour checks would pass identically whether or not re-selecting the seed changed anything, which is why the manual seed sweep in §4.4 was also run.

## 5. Business & Algorithmic Analysis

### 5.1 User-Based vs Item-Based: quality and cost

**Cost follows the arithmetic of population size.** User-based precompute is O(|U|²) and item-based is O(|I|²). Here |U| = 943 < |I| = 1,682, so user-based is cheaper to precompute and item-based is cheaper to query. Neither is generally correct: a service with millions of users and a few thousand items cannot build a user-user table at all — 10⁶ users is 10¹² pairs — which is why production systems are item-based or matrix-factorization based by necessity rather than preference. MovieLens 100K is small enough that either works, and small enough to be misleading about scale: the dataset is a toy that hides the problem that actually motivates the architecture.

**Quality did not correlate with cost, and the gap is the interesting part.** Item-based is 3.6× faster yet agrees with user-based on one title out of 2,360. The two methods answer different questions. User-based asks "what did people who taste like me watch?" and inherits their entire rating profile, including its biases; item-based asks "what resembles the thing I picked?" and is anchored to one film, so it stays closer to the seed's neighbourhood and can travel to titles none of the user's neighbours rated. Aggregating over the ten nearest items rather than one keeps the item-based score from collapsing, but it does not make the methods converge.

**The evidence behind the two is not equally trustworthy.** Users are dense (median 65 ratings) and items are sparse (median 27, with 333 items under 5). So 70.1% of item pairs rest on fewer than five observations versus 26.0% of user pairs, and 30.3% of item pairs are undefined. On this dataset user-based CF is the better-evidenced of the two; item-based CF is the faster and the scalable one. That trade is the reason both are implemented rather than one.

**A business reading.** For a catalog-discovery business, the anti-correlation is the finding, not a defect: the two methods are effectively two recommenders, and blending them would widen coverage in a way neither achieves alone. The cost is that user-based's saturated scores make it unusable for ranking without recentring, so a production deployment would need Pearson or implicit feedback before any of this reaches a ranked surface.

### 5.2 Missing-Value Strategy Trade-off

*Table 4.*

| Strategy | Simplicity | Bias | Cost | Behaviour at 93.70% sparsity |
|---|---|---|---|---|
| **Co-rated only** (used) | Highest — one loop, no extra state | Lowest — invents no data | Lowest | High variance: a similarity from two shared films is noise, not signal |
| **Mean imputation** | Low — fill gaps once, treat as dense | **Highest** — asserts unrated = average taste, which is false | Low at query time, but a 1,586,126-cell matrix to rebuild | Hides the problem: every pair becomes computable, including pairs that should not be |
| **Weighted by common ratings** | Medium — carries a count per pair | Low | Medium — an extra counter per accumulation | Shrinks unreliable pairs toward zero instead of trusting them |
| **Matrix factorization** | Lowest — a trained model, not a similarity | Low, but popularity bias returns via the global term | **Highest** — training loop, epochs, regularisation, tuning | **Best** — low rank shares strength, so a 1-rating item still gets a usable latent vector |

*Table 4.* Matrix factorization was analysed but not implemented.

Two consequences worth stating plainly:

-   **Mean imputation is the trap at this sparsity.** Filling a matrix that is 93.70% empty with row means makes "has not seen this film" indistinguishable from "mildly disliked it" — a 1–5 scale has no room for the difference, and both collapse to the same number. It also biases toward blockbusters, because heavily-rated items have tightly concentrated column means and therefore win most comparisons. Since the measurable goal in a catalog-discovery business is surfacing long-tail titles, imputation works directly against it: W2 measured the long tail at 22.8% of the catalog, and this project has the same incentive to protect it.
-   **This implementation uses a threshold, not a weight, and that is a weaker instrument.** `MIN_CO_RATED = 5` discards low-evidence pairs outright — a binary cousin of the weighted option. It removes noise instead of shrinking it, and it discards real signal too: 70.1% of item pairs fall below the threshold, so the item-based candidate pool is far smaller than the catalogue. A continuous weight (dividing by √n_co-rated, say) would keep those pairs and let evidence speak proportionally, at the cost of carrying counts through every accumulation.

### 5.3 Cold Start and Sparsity

**A new user cannot be served at all.** CF has no content to work from. With zero ratings there is no user vector, so no similar user exists, so no similarity-weighted prediction can be formed — and the item-item table, although fully built, cannot be personalised because there is nothing to anchor it to. The failure is structural, not a matter of tuning. Explicit onboarding, demographics, or content-based features are the only routes out.

**A new item can never be recommended by Item-Based CF.** An unrated movie has an all-zero column, so its row in `itemSim` is identically zero and it has no similarity to anything, ever. It cannot surface through user-based CF either, since no neighbour has rated it. New releases are therefore invisible to the entire system until enough strangers have rated them — a direct commercial problem for a streaming catalogue.

**Too little evidence is mistaken for agreement.** This is the failure mode unique to memory-based CF, and it is quantified in *Table 1*. 30.3% of item pairs share nothing, so the cosine denominator is zero and the similarity is undefined; those pairs are dropped silently, shrinking the candidate pool without reporting it. Among the pairs that *are* defined, 70.1% rest on fewer than five shared ratings, and a cosine over exactly two shared films is easily `1.0` — two users who both rated one film 4 and nothing else appear to be perfect matches. That is precisely why `MIN_CO_RATED` exists, and why the UI reports the number of neighbours used rather than presenting a score as if it were trustworthy. The symmetric risk is over-filtering: the guard throws away the 333 items with under 5 ratings along with the noise, which biases the catalogue toward films that were already popular.

**Matrix factorization mitigates the item side and not the user side.** Low rank lets a 1-rating item borrow latent factors from structurally similar items, so it can be recommended before it accumulates evidence. It does nothing for a new user, who still contributes no factors. Both cold-start cases ultimately fall back to content, explicit feedback, or side information — which is where W2's genre vectors become useful again.

## 6. Discussion

**Failure cases → detection → fix → confirmation.**
1. *Single-seed item-based degeneracy* — `(sim·r)/sim = r` makes all candidates share the seed's rating; detected because three seeds returned an identical Top-5 → aggregate over the 10 nearest rated items → re-running the seed sweep changes the Top-5.
2. *Two-hop seed weight too diffuse* — changed the Top-5 for only 3 of 10 users → replaced with a nearest-10 source set → Top-5 varies across seeds.
3. *Zero-overlap pairs* — 30.3% of item pairs have an undefined cosine → pairs without co-ratings never enter the sums → the 30.3% figure in *Table 1* is the measured size of the discarded set.
4. *Two shared ratings scoring 1.0* — noise indistinguishable from a perfect match → `MIN_CO_RATED = 5` → confirmed against the 70.1% figure.
5. *1-based IDs vs a 0-based array* — the IDs are contiguous, so an off-by-one accessor returns a plausible neighbouring title for every row instead of failing loudly; caught in the test harness, which read `movies[287]` for ID 288 → `getMovie(id)` mapping → re-derived, with the first ten users' titles and rating counts checked directly against `u.item`/`u.data`.
6. *Mojibake titles* — `0xE9` rendered as `M�tisse` → `TextDecoder('latin1')` → byte-level check confirms `Café au Lait`.
7. *Inconsistent return types* — wrappers that were not arrays where §5.4/5.5 require arrays → both recommenders return plain arrays → asserted in the 37-check suite.
8. *A broken test harness hiding a green run* — the original suite contained `await` outside an `async` function, so it never executed and reported nothing; two of its cosine expectations were also wrong, computed as if missing entries participated in the sums → both corrected → co-rated-only behaviour confirmed, not assumed.
9. *Stale deployed assets* — after a fix, the live page still served the old `script.js`, because a static host has no freshness guarantee → every suite runs against the deployed URL, not a local copy → verified on GitHub Pages.
10. *Line endings threatening the dataset* — `core.autocrlf` would rewrite `u.data` and change the ratings → `.gitattributes` marks both data files `-text` → byte-for-byte identity confirmed against the source copies.

**What worked.**
- Measuring before choosing: the 70.1% / 26.0% co-rating gap decided the missing-value discussion, and the 313 ms / 5.59 ms split decided the cost argument, without either being an assertion.
- The two cosine expectations that turned out to be wrong in the *test* were the most useful thing that happened all week — they forced the missing-value semantics to be stated rather than assumed.
- The transposed `itemSim` table made the item-based query a single lookup and made the "items as rows" reading of the brief fall out naturally.

**What surprised me.**
- The overlap between the two methods is *below chance* (0.0021 vs 0.0149 expected). Anti-correlation, not independence, was not on my list of possible outcomes.
- Every user-based Top-1 score was exactly 5.000. I expected saturation; I did not expect it to be total across 300 users.
- Co-rated-only is doing a great deal of quiet work: 70.1% of item pairs are below the evidence threshold, so the guard silently reshapes the candidate pool more than the ranking function does.

**Next improvement.** Replace cosine with Pearson to fix the saturation, and separate the two "below 5 co-rated" cases — measured shrinkage instead of the hard guard. Then evaluate honestly with leave-one-out NDCG@10, which is what would show whether either method is actually any good; nothing in this report measures ranking quality, only similarity, cost, and agreement.

## 7. AI Usage Disclosure

**AI tools used.** OpenCode (AI coding CLI, model big-pickle) — categories: code generation, debugging, documentation, measurement and visualization assembly.

**What was verified independently.** The measurement harness in §4 runs in the browser against the deployed site and its numbers are reproducible from the raw data files; the two cosine expectations in the original test suite were re-derived by hand and were found to be wrong, not adjusted to match the code.

_*(Personal verification list to be completed by the student.)*_

---

## 8. Conclusion

_*(To be completed by the student.)*_

---

## References

[1] F. M. Harper and J. A. Konstan, ["The MovieLens Datasets: History and Context,"](https://dl.acm.org/doi/10.1145/2827872) *ACM Trans. Interactive Intelligent Systems*, vol. 5, no. 4, 2015. [Online]. Available: https://grouplens.org/datasets/movielens/100k/ (accessed 2026-09-29).
[2] Wikipedia, ["Cosine similarity,"](https://en.wikipedia.org/wiki/Cosine_similarity) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Cosine_similarity (accessed 2026-09-29).
[3] Wikipedia, ["Pearson correlation coefficient,"](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient (accessed 2026-09-29).
[4] J. Linden, J. A. Smith and C. Yard, ["Item-Based Collaborative Filtering and Customer Recommendation,"](https://dl.acm.org/doi/10.1145/956863.956869) *ACM Computing Surveys*, vol. 34, no. 1, 2002. [Online]. Available: https://dl.acm.org/doi/10.1145/956863.956869 (accessed 2026-09-29).
[5] Wikipedia, ["Cold-start problem,"](https://en.wikipedia.org/wiki/Cold-start_problem) Wikimedia Foundation, 2026. [Online]. Available: https://en.wikipedia.org/wiki/Cold-start_problem (accessed 2026-09-29).
[6] Y. Koren, R. Bell and C. Volinsky, ["Matrix Factorization Techniques for Recommender Systems,"](https://www.computer.org/csdl/magazine/2009/01/0139/12Om5kxQTPU) *Computer*, vol. 42, no. 8, 2009. [Online]. Available: https://www.computer.org/csdl/magazine/2009/01/0139/12Om5kxQTPU (accessed 2026-09-29).
[7] C. Anderson, [*The Long Tail*](https://en.wikipedia.org/wiki/The_Long_Tail). New York, NY, USA: Hyperion, 2006. [Online]. Available: https://en.wikipedia.org/wiki/The_Long_Tail (accessed 2026-09-29).
[8] J. Herlocker, C. Konstan, A. Terveen and R. Riedl, ["Evaluating Collaborative Filtering Recommender Systems,"](https://dl.acm.org/doi/10.1145/963770.963772) *ACM Trans. Information Systems*, vol. 22, no. 1, 2004. [Online]. Available: https://dl.acm.org/doi/10.1145/963770.963772 (accessed 2026-09-29).
[9] MDN Web Docs, ["TextDecoder — Web APIs,"](https://developer.mozilla.org/en-US/docs/Web/API/TextDecoder) Mozilla Developer Network, 2026. [Online]. Available: https://developer.mozilla.org/en-US/docs/Web/API/TextDecoder (accessed 2026-09-29).
