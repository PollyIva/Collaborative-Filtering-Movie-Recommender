// ============================================================================
// script.js — UI + collaborative-filtering logic (MovieLens 100K)
//
// Contains everything that "thinks": the cosine similarity function, the
// User-Based and Item-Based recommendation routines, and the DOM rendering.
// All raw data comes from data.js (movies, ratings, ratingMatrix, ratedMask).
// ============================================================================

// ============================================================================
// MISSING VALUE STRATEGY  (chosen once, applied everywhere)
// ----------------------------------------------------------------------------
// Strategy: CO-RATED ITEMS ONLY — missing entries are ignored, never imputed.
//
// Why this one: the rating matrix is ~93.7% empty (100,000 ratings in a
// 943 x 1,682 grid). Filling the gaps with zeros, a global mean, or a row
// average fabricates data that the algorithm then treats as real evidence.
// The MovieLens scale is strictly 1..5, so a 0 can never be a genuine rating
// and "not rated" is unambiguous. Restricting every cosine computation to the
// movies two users (or two movies) both rated therefore measures agreement
// only where evidence exists.
//
// Consequence to be aware of: similarity is estimated from the co-rated subset
// alone, so pairs with very little overlap are unreliable. The MIN_CO_RATED
// guard below exists precisely to stop those noisy pairs from winning.
// Mean imputation and co-rated-count weighting are deliberately NOT mixed in.
// ============================================================================

// --- Tunable parameters -----------------------------------------------------
const TOP_K = 5;             // size of the recommendation list
const N_NEIGHBOURS = 20;     // neighbours kept for User-Based CF
const MIN_CO_RATED = 5;      // minimum co-rated items for a trustworthy similarity
const SEED_NEIGHBOURS = 10; // rated movies closest to the chosen Item-Based seed that count

// --- Cached state -----------------------------------------------------------
let itemSim = null;          // item x item similarity, built once after loading
let isReady = false;         // true once data AND similarities are available
// The prompt fixes the return type of both recommenders to an array of
// { title, score }, so the extra context the UI wants is published here instead
// of being smuggled out in a wrapper object.
let lastNeighbours = [];     // neighbours used by the last User-Based call
let lastSeedMovie = null;    // seed used by the last Item-Based call

// ============================================================================
// INITIALISATION
// ============================================================================

// Runs after the DOM is ready: load the data, fill the dropdown, precompute the
// item-item similarities, then unlock the button.
window.onload = async function () {
    try {
        setStatus('Loading MovieLens data…');

        // Data loading lives in data.js.
        await loadData();

        // Populate the selector as soon as the data exists, exactly as
        // specified. The button stays disabled until the precompute below
        // finishes, so a premature click cannot reach a half-built table.
        populateUserDropdown();

        // The movie list belongs to the active user, and the user dropdown
        // defaults to its first option, so build it for that user too.
        populateMovieDropdown(getSelectedUserId());
        updateSelectionSummary();

        setStatus(`Loaded ${numMovies} movies and ${ratings.length} ratings ` +
                  `from ${numUsers} users. Building item-item similarities…`);

        // Precomputing here (rather than on first click) keeps every later
        // query instant. It is chunked so the page never freezes.
        await buildItemSimilarities();

        setStatus(`Ready — pick a user and press "Get Recommendations". ` +
                  `${numUsers} users, ${numMovies} movies loaded.`);
        isReady = true;

        document.getElementById('recommend-btn').disabled = false;
    } catch (error) {
        // loadData() already reported the problem in the UI.
        setStatus(`Error: ${error.message}`, true);
        isReady = false;
    }
};

// --------------------------------------------------------------------------
// setStatus(message, isError)
// --------------------------------------------------------------------------
function setStatus(message, isError = false) {
    const el = document.getElementById('status');
    el.textContent = message;
    el.classList.toggle('error', isError);
}

// --------------------------------------------------------------------------
// populateUserDropdown()
// One option per user id, 1..numUsers, with the value set to the integer id.
// --------------------------------------------------------------------------
function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';

    for (let id = 1; id <= numUsers; id++) {
        const option = document.createElement('option');
        option.value = id;
        const count = countRatings(id);
        option.textContent = `User ${id} (${count} ratings)`;
        userSelect.appendChild(option);
    }

    userSelect.disabled = false;
    userSelect.addEventListener('change', onUserChanged);
}

// --------------------------------------------------------------------------
// populateMovieDropdown(userId)
// Fills #movie-select with the movies THAT user has actually rated, best first.
//
    // These are the only sensible seeds for Item-Based CF here: the algorithm
    // aggregates similarity away from "the movie this user chose", and a movie
    // the user never rated has no place in their history to aggregate from.
    // The list is sorted best-rated first, but every rated movie is selectable
    // — the rating is used as a weight, so a 1/5 seed simply scores lower.

// --------------------------------------------------------------------------
function populateMovieDropdown(userId) {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';

    if (!userId || userId < 1 || userId > numUsers) {
        movieSelect.disabled = true;
        updateSelectionSummary();
        return;
    }

    const rated = Array.from(userItems[userId])
        .filter(([, rating]) => rating > 0)
        .sort((a, b) => b[1] - a[1] || a[0] - b[0]);

    for (const [itemId, rating] of rated) {
        const option = document.createElement('option');
        option.value = itemId;
        option.textContent = `${getMovieTitle(itemId)} — rated ${rating}/5`;
        movieSelect.appendChild(option);
    }

    // Nothing rated means nothing to seed from.
    movieSelect.disabled = rated.length === 0;
    updateSelectionSummary();
}

// --------------------------------------------------------------------------
// onUserChanged()
// The movie list depends on the user, so it is rebuilt on every change.
// --------------------------------------------------------------------------
function onUserChanged() {
    populateMovieDropdown(getSelectedUserId());
}

// --------------------------------------------------------------------------
// getSelectedUserId() / getSelectedMovieId()
// The two dropdowns as integers. Empty selection reads as 0.
// --------------------------------------------------------------------------
function getSelectedUserId() {
    return parseInt(document.getElementById('user-select').value, 10) || 0;
}

function getSelectedMovieId() {
    return parseInt(document.getElementById('movie-select').value, 10) || 0;
}

// --------------------------------------------------------------------------
// updateSelectionSummary()
// "Prints" who is selected and which movie is driving the Item-Based panel,
// both into the page and into the browser console.
// --------------------------------------------------------------------------
function updateSelectionSummary() {
    const el = document.getElementById('selection');
    const userId = getSelectedUserId();
    const itemId = getSelectedMovieId();

    if (!userId) {
        el.textContent = 'Selected user: none — choose a user to begin.';
        el.classList.add('muted');
        console.log('[selection] user: none, movie: none');
        return;
    }

    const userPart = `Selected user: User ${userId} (${countRatings(userId)} ratings)`;

    if (!itemId) {
        el.textContent = `${userPart} — no movie selected.`;
        el.classList.add('muted');
        console.log(`[selection] user: User ${userId}, movie: none`);
        return;
    }

    const title = getMovieTitle(itemId);
    const rating = ratingMatrix[userId][itemId];

    el.textContent = `${userPart} | Selected movie: ${title} (rated ${rating}/5)`;
    el.classList.remove('muted');

    console.log(`[selection] user: User ${userId}`);
    console.log(`[selection] movie: ${title}`);
}

// --------------------------------------------------------------------------
// countRatings(userId)
// Number of movies the user actually rated — used for the dropdown label and
// to give a clear message for users with too little history.
// --------------------------------------------------------------------------
function countRatings(userId) {
    return userItems[userId].length;
}

// ============================================================================
// SIMILARITY
// ============================================================================

// --------------------------------------------------------------------------
// cosineSimilarity(a, b)
//
// Cosine similarity between two equally long rating vectors, computed over
// co-rated (non-zero) entries only:
//
//     cos = Σ a[i]·b[i]  /  sqrt( Σ a[i]² · Σ b[i]² )
//
// where both sums run over the indices where a[i] != 0 AND b[i] != 0.
//
// Missing-value note: an entry equal to 0 means "not rated", not "rated zero"
// (see the strategy note at the top of this file), so a movie only one of the
// two parties has seen is skipped entirely rather than contributing a 0 that
// would deflate both the numerator and the norms.
//
// Returns 0 when the two vectors share no rated entries, or when either norm
// is zero — that guard keeps the division safe for degenerate pairs.
// --------------------------------------------------------------------------
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;

    for (let i = 0; i < a.length; i++) {
        const x = a[i];
        const y = b[i];
        if (x !== 0 && y !== 0) {
            dot += x * y;
            normA += x * x;
            normB += y * y;
        }
    }

    if (normA === 0 || normB === 0) return 0;

    return dot / Math.sqrt(normA * normB);
}

// --------------------------------------------------------------------------
// coRatedCount(a, b)
// How many positions both vectors rated. Used only to reject similarity
// estimates built from too little overlap (MIN_CO_RATED).
// --------------------------------------------------------------------------
function coRatedCount(a, b) {
    let n = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== 0 && b[i] !== 0) n++;
    }
    return n;
}

// ============================================================================
// PRECOMPUTATION — item-item similarity
// ============================================================================

// --------------------------------------------------------------------------
// buildItemSimilarities()
//
// Fills `itemSim` (a symmetric numMovies+1 square matrix) with the cosine
// similarity between every pair of movie rating columns, co-rated only.
//
// WHY THIS IS PRECOMPUTED: building the table the obvious way — for every one
// of the 1,413,721 item pairs, scan all 943 rating columns — costs ~1.33 billion
// full-column reads and locks the browser up. The loop below produces exactly
// the same numbers (it is the same co-rated cosine formula, just driven from an
// inverted index instead of scanning full columns) in 10,050,406 pair updates,
// once, for all users.
//
// For a fixed movie i it walks only the users who rated i, and for each of them
// only the movies that same user rated above i. Every (user, item-pair)
// combination is therefore touched exactly once, and dot/na/nb accumulate the
// three sums the co-rated cosine needs.
//
// The work is chunked with an await between batches so the UI stays responsive.
// --------------------------------------------------------------------------
// `minCoRated` is a parameter only so the experiments in experiments/ can
// measure the effect of the floor itself (passing 0 rebuilds the unguarded
// cache for the ablation). The app itself always calls it with the default.
async function buildItemSimilarities(minCoRated = MIN_CO_RATED) {
    const width = numMovies + 1;
    itemSim = new Float32Array(width * width);

    // Accumulators for the movie i currently being processed. These are tiny
    // (one entry per movie) and are reused for every i.
    const accDot = new Float64Array(width);
    const accNa = new Float64Array(width);
    const accNb = new Float64Array(width);
    const accCount = new Int32Array(width);
    const touched = new Int32Array(width);

    const CHUNK = 100; // movies processed before yielding to the browser

    for (let i = 1; i <= numMovies; i++) {
        let nTouched = 0;

        // Walk the users who rated movie i (the inverted index from data.js).
        for (const [userId, ratingI] of itemUsers[i]) {
            const naI = ratingI * ratingI;

            // Only movies j > i, so each pair is evaluated exactly once.
            for (const [itemId, ratingJ] of userItems[userId]) {
                if (itemId <= i) continue;

                if (accCount[itemId] === 0) touched[nTouched++] = itemId;

                accDot[itemId] += ratingI * ratingJ;
                accNa[itemId] += naI;
                accNb[itemId] += ratingJ * ratingJ;
                accCount[itemId]++;
            }
        }

        // Turn the accumulators into a similarity and store it symmetrically.
        const base = i * width;
        for (let k = 0; k < nTouched; k++) {
            const j = touched[k];

            // Same evidence floor as getUserBasedRecommendations. Without it
            // accCount is computed and thrown away, and the cache fills up with
            // cosines taken from a single shared rating — two users who both
            // scored one film 4 store similarity 1.000, which then outranks
            // every properly-measured pair. Measured on user 1, the unguarded
            // cache put four films at cos = 1.0000 from ONE co-rater above all
            // 0.99 neighbours built from 5+ co-raters.
            //
            // Measured on held-out NDCG@10 over all 943 users
            // (experiments/run.py): 0.0013 with the floor, 0.0009 without, and
            // 0.0100 for a random list. So the guard is NOT what makes Item-Based
            // rank badly — removing it changes nothing meaningful. The real cause
            // is that cosine between 1..5 ratings is near 1.0 for almost any
            // pair sharing five films, so this table orders candidates by rater
            // count rather than by taste. The floor stays because a similarity
            // estimated from one shared rating is not evidence, not because it
            // measurably helps the ranking.
            // Pairs below the floor stay 0, which getItemBasedRecommendations
            // already skips via its `sim <= 0` test.
            if (accCount[j] < minCoRated) continue;

            const denom = accNa[j] * accNb[j];

            // denom === 0 means the pair had no co-rated user with a non-zero
            // rating — the same zero-denominator guard as cosineSimilarity.
            const sim = denom > 0 ? accDot[j] / Math.sqrt(denom) : 0;

            itemSim[base + j] = sim;
            itemSim[j * width + i] = sim;
        }

        // Reset only the entries this movie touched.
        for (let k = 0; k < nTouched; k++) {
            const j = touched[k];
            accDot[j] = 0; accNa[j] = 0; accNb[j] = 0; accCount[j] = 0;
        }

        // Yield periodically so the "Building similarities…" message can paint.
        if (i % CHUNK === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
}

// ============================================================================
// USER-BASED COLLABORATIVE FILTERING
// ============================================================================

// --------------------------------------------------------------------------
// getUserBasedRecommendations(activeUserId, topK)
//
//   Step 1. Cosine similarity between the active user and every other user.
//   Step 2. Keep the N_NEIGHBOURS most similar users with positive similarity
//           (and enough co-rated movies to be meaningful).
//   Step 3. For each unseen movie, predict the similarity-weighted average of
//           the neighbours' ratings.
//   Step 4. Sort by predicted score, descending, and return the top topK.
//
// Returns an ARRAY of { title, score }, sorted by score descending (prompt 5.4).
// An empty array is returned on every failure path, never null or a wrapper.
// The neighbour list is also published on the module as `lastNeighbours` so the
// status line can report it without breaking the required return type.
// --------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = TOP_K) {
    lastNeighbours = [];

    if (!activeUserId || activeUserId < 1 || activeUserId > numUsers) {
        return [];
    }

    const activeRow = ratingMatrix[activeUserId];

    // --- Step 1 + 2: find the most similar users ---------------------------
    const neighbours = [];

    for (let other = 1; other <= numUsers; other++) {
        if (other === activeUserId) continue;

        const similarity = cosineSimilarity(activeRow, ratingMatrix[other]);
        if (similarity <= 0) continue;

        // A high cosine computed from one or two shared movies is noise, not
        // agreement — for example two users who both gave one film a 4 score
        // 1.0 apart get cosine 1.0. Requiring a minimum overlap keeps those
        // pairs from crowding out genuinely similar users.
        if (coRatedCount(activeRow, ratingMatrix[other]) < MIN_CO_RATED) continue;

        neighbours.push({ userId: other, similarity });
    }

    // Highest similarity first; the id breaks ties so the output is stable.
    neighbours.sort((a, b) =>
        b.similarity - a.similarity || a.userId - b.userId);

    const top = neighbours.slice(0, N_NEIGHBOURS);
    lastNeighbours = top;

    if (top.length === 0) {
        return [];
    }

    // --- Step 3: predict a score for every movie the user has not seen -----
    const activeMask = ratedMask[activeUserId];
    const candidates = [];

    for (let itemId = 1; itemId <= numMovies; itemId++) {
        if (activeMask[itemId]) continue; // already seen — cannot recommend it

        let weightedSum = 0;
        let similaritySum = 0;

        for (const { userId, similarity } of top) {
            if (ratedMask[userId][itemId]) {
                weightedSum += similarity * ratingMatrix[userId][itemId];
                similaritySum += similarity;
            }
        }

        // Only movies at least one neighbour rated can be scored at all.
        if (similaritySum > 0) {
            candidates.push({
                itemId,
                title: getMovieTitle(itemId),
                score: weightedSum / similaritySum
            });
        }
    }

    // --- Step 4: rank and cut ---------------------------------------------
    candidates.sort((a, b) => b.score - a.score || a.itemId - b.itemId);

    // Prompt 5.4: return an array of { title, score }.
    return candidates.slice(0, topK).map(({ title, score }) => ({ title, score }));
}

// ============================================================================
// ITEM-BASED COLLABORATIVE FILTERING
// ============================================================================

// --------------------------------------------------------------------------
// getItemBasedRecommendations(activeUserId, topK, seedItemId)
//
//   Step 1. Item-to-item cosine similarity between every movie's rating column
//           and every other movie's column — precomputed into `itemSim` by
//           buildItemSimilarities() using the same co-rated cosine formula.
//   Step 2. For each unseen movie, aggregate the similarities coming from the
//           movies this user did rate, weighted by the rating they gave.
//   Step 3. Sort by the aggregated score, descending, and return top topK.
//
// score(j) = Σ_i w_i·sim(i,j)·r_ui  /  Σ_i w_i·sim(i,j)   over the user's rated i
//
// Dividing by the similarity sum turns the raw weighted total into a weighted
// average, which keeps the result on the same 1..5 scale as User-Based CF and
// makes the two panels directly comparable.
//
// seedItemId (an addition on top of the prompt, driven by #movie-select):
// the chosen movie anchors the search. Rather than replacing the user's history
// with that one movie, the chosen movie decides *which* of the user's rated
// movies get to contribute: the SEED_NEIGHBOURS rated movies closest to the
// seed. With no seed (or an invalid one) every rated movie contributes, which
// is exactly the plain basket aggregation the prompt asks for.
//
// Why not simply seed from the chosen movie alone: the score would collapse to
// (sim·r)/(sim) = r, a constant for every candidate, and the "ranking" would
// degenerate to movie-ID order. Keeping the basket is both spec-compliant and
// non-degenerate.
//
// Returns an ARRAY of { title, score } (prompt 5.5).
// --------------------------------------------------------------------------
function getItemBasedRecommendations(activeUserId, topK = TOP_K, seedItemId = 0) {
    if (!activeUserId || activeUserId < 1 || activeUserId > numUsers) return [];
    if (!itemSim) return [];

    const width = numMovies + 1;
    const activeMask = ratedMask[activeUserId];
    const activeRow = ratingMatrix[activeUserId];

    // An explicitly chosen seed wins, but only if the user really rated it.
    let seed = null;
    if (seedItemId && activeMask[seedItemId] && activeRow[seedItemId] > 0) {
        seed = { itemId: seedItemId, rating: activeRow[seedItemId], title: getMovieTitle(seedItemId) };
    } else {
        seed = findTopRatedMovie(activeUserId, activeRow);
    }
    lastSeedMovie = seed;

    const rated = userItems[activeUserId].filter(([, r]) => r > 0);
    if (rated.length === 0) return [];

    // --- Step 1 + 2: aggregate similarity from the user's rated movies ------
    // With a seed, only the rated movies that actually resemble the chosen one
    // are allowed to contribute. A flat two-hop weight was tried first and
    // proved too diffuse: cosine over 1..5 ratings is high for almost every
    // pair, so the weights came out nearly uniform and the chosen seed barely
    // moved the ranking (it changed the Top-5 for only 3 of 10 sampled users).
    // Restricting the basket to the seed's strongest matches makes the choice
    // decisive while still aggregating over the user's own ratings, as the
    // prompt requires.
    let sources = rated;

    if (seed) {
        const seedBase = seed.itemId * width;

        sources = rated
            .map(([itemId, ratingI]) => ({
                itemId,
                ratingI,
                simToSeed: itemSim[seedBase + itemId]
            }))
            .filter(s => s.simToSeed > 0)
            .sort((a, b) => b.simToSeed - a.simToSeed || a.itemId - b.itemId)
            .slice(0, SEED_NEIGHBOURS);

        if (sources.length === 0) return [];
    }

    const weighted = new Float64Array(numMovies + 1);
    const similarityTotal = new Float64Array(numMovies + 1);

    for (const { itemId, ratingI } of sources) {
        const base = itemId * width;

        for (let candidate = 1; candidate <= numMovies; candidate++) {
            if (candidate === itemId) continue;
            if (activeMask[candidate]) continue; // already rated

            const sim = itemSim[base + candidate];

            // Only positive similarity contributes. Negative similarity would
            // push a movie down instead of up, so it is skipped.
            if (sim <= 0) continue;

            weighted[candidate] += sim * ratingI;
            similarityTotal[candidate] += sim;
        }
    }

    // --- Step 3: rank and cut ---------------------------------------------
    const candidates = [];

    for (let candidate = 1; candidate <= numMovies; candidate++) {
        if (similarityTotal[candidate] > 0) {
            candidates.push({
                itemId: candidate,
                title: getMovieTitle(candidate),
                score: weighted[candidate] / similarityTotal[candidate]
            });
        }
    }

    candidates.sort((a, b) => b.score - a.score || a.itemId - b.itemId);

    // Prompt 5.5: return an array of { title, score }.
    return candidates.slice(0, topK).map(({ title, score }) => ({ title, score }));
}

// --------------------------------------------------------------------------
// findTopRatedMovie(activeUserId, activeRow)
// The highest-rated movie the user gave, for the explanatory sentence.
// Returns null when the user has rated nothing.
// --------------------------------------------------------------------------
function findTopRatedMovie(activeUserId, activeRow) {
    let best = null;

    for (const [itemId, ratingI] of userItems[activeUserId]) {
        if (!best || ratingI > best.rating || (ratingI === best.rating && itemId < best.itemId)) {
            best = { itemId, rating: ratingI, title: getMovieTitle(itemId) };
        }
    }

    return best;
}

// --------------------------------------------------------------------------
// getMovieTitle(itemId)
// Looks the title up in `movies`, falling back to the raw id if the catalogue
// is missing an entry.
// --------------------------------------------------------------------------
function getMovieTitle(itemId) {
    const movie = movies.find(m => m.id === itemId);
    return movie ? movie.title : `Movie #${itemId}`;
}

// ============================================================================
// RENDERING
// ============================================================================

// --------------------------------------------------------------------------
// getRecommendations()
// Called by the button. Runs both algorithms for the selected user and
// renders each result into its own panel.
// --------------------------------------------------------------------------
function getRecommendations() {
    if (!isReady) {
        setStatus('Still loading the dataset — please wait.', true);
        return;
    }

    const userId = getSelectedUserId();
    const itemId = getSelectedMovieId();

    if (!userId || Number.isNaN(userId)) {
        setStatus('Please select a user first.', true);
        return;
    }

    updateSelectionSummary();

    const ratedCount = countRatings(userId);

    // Both algorithms run on every call, even when one of them cannot produce
    // a list, so the two panels always describe the same user.
    const userBased = getUserBasedRecommendations(userId, TOP_K);
    const itemBased = getItemBasedRecommendations(userId, TOP_K, itemId);

    const userBasedEmpty = userBased.length === 0;
    const itemBasedEmpty = itemBased.length === 0;

    // The status line describes what actually happened rather than assuming
    // failure: the two approaches have different data requirements. Item-Based
    // CF can still work from two rated movies, while User-Based CF needs enough
    // overlap to find genuinely similar neighbours.
    if (userBasedEmpty && itemBasedEmpty) {
        setStatus(`User ${userId} has only ${ratedCount} rating(s). ` +
                  `There is not enough history for either approach — please pick ` +
                  `someone with at least ${MIN_CO_RATED} ratings.`, true);
    } else if (userBasedEmpty || itemBasedEmpty) {
        const failed = userBasedEmpty ? 'User-Based' : 'Item-Based';
        const worked = userBasedEmpty ? 'Item-Based' : 'User-Based';
        setStatus(`User ${userId} has rated ${ratedCount} movies. ` +
                  `${worked} CF produced a Top-5, but ${failed} CF could not — ` +
                  `this user has too little history for it.`);
    } else {
        const seedNote = lastSeedMovie
            ? `anchored on "${lastSeedMovie.title}"`
            : 'with no usable seed movie';
        setStatus(`User ${userId} has rated ${ratedCount} movies. ` +
                  `User-Based CF compared them with ${lastNeighbours.length} ` +
                  `most similar users; Item-Based CF ran ${seedNote}.`);
    }

    renderUserBased(userId, userBased, ratedCount);
    renderItemBased(userId, itemBased, ratedCount);
}

// --------------------------------------------------------------------------
// renderUserBased(userId, recommendations, ratedCount)
//
// `recommendations` is the array returned by getUserBasedRecommendations; the
// neighbours that produced it are read from `lastNeighbours`.
// --------------------------------------------------------------------------
function renderUserBased(userId, result, ratedCount) {
    const panel = document.getElementById('user-based-result');
    const why = panel.querySelector('.panel-why');
    const list = panel.querySelector('.rec-list');

    const neighbours = lastNeighbours;

    if (result.length === 0) {
        why.textContent = 'Because you are similar to other users, we recommend:';
        list.replaceChildren(buildEmptyMessage(
            ratedCount < MIN_CO_RATED
                ? `User ${userId} has rated only ${ratedCount} movie(s), which is not ` +
                  'enough to find similar users.'
                : `No recommendation could be scored for User ${userId}: none of the ` +
                  'movies they have not seen were rated by a sufficiently similar user.'));
        return;
    }

    // Name the most influential neighbour to make the reasoning concrete.
    const topNeighbour = neighbours[0];
    why.textContent =
        `Because you are similar to other users (most similar: User ${topNeighbour.userId}, ` +
        `similarity ${topNeighbour.similarity.toFixed(3)}), we recommend:`;

    list.replaceChildren(...result.map((rec, index) =>
        buildListItem(index + 1, rec.title, rec.score.toFixed(2))));
}

// --------------------------------------------------------------------------
// renderItemBased(userId, recommendations, ratedCount)
//
// `recommendations` is the array returned by getItemBasedRecommendations; the
// seed that produced it is read from `lastSeedMovie`.
// --------------------------------------------------------------------------
function renderItemBased(userId, result, ratedCount) {

    const panel = document.getElementById('item-based-result');
    const why = panel.querySelector('.panel-why');
    const list = panel.querySelector('.rec-list');

    if (result.length === 0) {
        why.textContent = 'Because you liked certain movies, we recommend:';
        list.replaceChildren(buildEmptyMessage(
            ratedCount < MIN_CO_RATED
                ? `User ${userId} has rated only ${ratedCount} movie(s), which is not ` +
                  'enough to find similar movies.'
                : `No recommendation could be scored for User ${userId}: none of the ` +
                  'unseen movies share a positive similarity with the ones already rated.'));
        return;
    }

    const seedMovie = lastSeedMovie;
    why.textContent = seedMovie
        ? `Because you liked "${seedMovie.title}" (your rating: ` +
          `${seedMovie.rating}/5), we recommend:`
        : 'Because you liked certain movies, we recommend:';

    list.replaceChildren(...result.map((rec, index) =>
        buildListItem(index + 1, rec.title, rec.score.toFixed(2))));
}

// --------------------------------------------------------------------------
// buildListItem(rank, title, score)
// One row of a recommendation list.
// --------------------------------------------------------------------------
function buildListItem(rank, title, score) {
    const li = document.createElement('li');

    const titleEl = document.createElement('span');
    titleEl.className = 'rec-title';
    titleEl.textContent = title;

    const scoreEl = document.createElement('span');
    scoreEl.className = 'rec-score';
    scoreEl.textContent = score;

    li.append(titleEl, scoreEl);
    return li;
}

// --------------------------------------------------------------------------
// buildEmptyMessage(text)
// The graceful "nothing to show" state required for users with too few ratings.
// --------------------------------------------------------------------------
function buildEmptyMessage(text) {
    const p = document.createElement('p');
    p.className = 'rec-empty';
    p.textContent = text;
    return p;
}
