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

// --- Cached state -----------------------------------------------------------
let itemSim = null;          // item x item similarity, built once after loading
let isReady = false;         // true once data AND similarities are available

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
// WHY THIS IS PRECOMPUTED: the obvious implementation — for each of the
// user's rated movies, call cosineSimilarity against all 1,682 movies — costs
// 1.17 billion cell reads for a heavy MovieLens user and locks the browser up.
// The loop below produces exactly the same numbers (it is the same co-rated
// cosine formula, just driven from an inverted index instead of scanning full
// columns) in about 10 million operations, once, for all users.
//
// For a fixed movie i it walks only the users who rated i, and for each of them
// only the movies that same user rated above i. Every (user, item-pair)
// combination is therefore touched exactly once, and dot/na/nb accumulate the
// three sums the co-rated cosine needs.
//
// The work is chunked with an await between batches so the UI stays responsive.
// --------------------------------------------------------------------------
async function buildItemSimilarities() {
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
// Returns { recommendations: [{ title, score }, ...], neighbours: [{ userId, similarity }] }.
// The wrapper is always returned — never a bare array — so callers can rely on
// the same shape on every path, including the guards below.
// --------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = TOP_K) {
    const empty = { recommendations: [], neighbours: [] };

    if (!activeUserId || activeUserId < 1 || activeUserId > numUsers) {
        return empty;
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

    if (top.length === 0) {
        return empty;
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

    return {
        recommendations: candidates.slice(0, topK),
        neighbours: top
    };
}

// ============================================================================
// ITEM-BASED COLLABORATIVE FILTERING
// ============================================================================

// --------------------------------------------------------------------------
// getItemBasedRecommendations(activeUserId, topK)
//
//   Step 1. Item-to-item cosine similarity between every movie's rating column
//           and every other movie's column — precomputed into `itemSim` by
//           buildItemSimilarities() using the same co-rated cosine formula.
//   Step 2. For each unseen movie, aggregate the similarities coming from the
//           movies this user did rate, weighted by the rating they gave.
//   Step 3. Sort by the aggregated score, descending, and return top topK.
//
// score(j) = Σ_i sim(i,j)·r_ui  /  Σ_i sim(i,j)   over the user's rated i
//
// Dividing by the similarity sum turns the raw weighted total into a weighted
// average, which keeps the result on the same 1..5 scale as User-Based CF and
// makes the two panels directly comparable.
//
// Returns { recommendations: [{ title, score }, ...], seedMovie: { title, rating } | null }.
// Like the user-based version, the wrapper is always returned so the shape is
// stable on every path.
// --------------------------------------------------------------------------
function getItemBasedRecommendations(activeUserId, topK = TOP_K) {
    const empty = { recommendations: [], seedMovie: null };

    if (!activeUserId || activeUserId < 1 || activeUserId > numUsers) return empty;
    if (!itemSim) return empty;

    const width = numMovies + 1;
    const activeMask = ratedMask[activeUserId];
    const activeRow = ratingMatrix[activeUserId];

    const weighted = new Float64Array(numMovies + 1);
    const similarityTotal = new Float64Array(numMovies + 1);

    // --- Step 1 + 2: accumulate similarity from every movie already seen ---
    for (const [itemId, ratingI] of userItems[activeUserId]) {
        if (ratingI <= 0) continue; // only positive ratings act as "likes"

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

    const recommendations = candidates.slice(0, topK);

    return {
        recommendations,
        // The movie the user rated highest — used for the "Because you liked…"
        // sentence, so the panel explains its own reasoning.
        seedMovie: findTopRatedMovie(activeUserId, activeRow)
    };
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

    const userSelect = document.getElementById('user-select');
    const userId = parseInt(userSelect.value, 10);

    if (!userId || Number.isNaN(userId)) {
        setStatus('Please select a user first.', true);
        return;
    }

    const ratedCount = countRatings(userId);

    // Both algorithms run on every call, even when one of them cannot produce
    // a list, so the two panels always describe the same user.
    const userBased = getUserBasedRecommendations(userId, TOP_K);
    const itemBased = getItemBasedRecommendations(userId, TOP_K);

    const userBasedEmpty = userBased.recommendations.length === 0;
    const itemBasedEmpty = itemBased.recommendations.length === 0;

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
        setStatus(`User ${userId} has rated ${ratedCount} movies. ` +
                  `User-Based CF compared them with ${userBased.neighbours.length} ` +
                  `most similar users; Item-Based CF used the movies already rated.`);
    }

    renderUserBased(userId, userBased, ratedCount);
    renderItemBased(userId, itemBased, ratedCount);
}

// --------------------------------------------------------------------------
// renderUserBased(userId, result, ratedCount)
// --------------------------------------------------------------------------
function renderUserBased(userId, result, ratedCount) {
    const panel = document.getElementById('user-based-result');
    const why = panel.querySelector('.panel-why');
    const list = panel.querySelector('.rec-list');

    const { recommendations, neighbours } = result;

    if (recommendations.length === 0) {
        why.textContent = 'Because you are similar to other users, we recommend:';
        list.replaceChildren(buildEmptyMessage(
            ratedCount < MIN_CO_RATED
                ? `User ${userId} has rated only ${ratedCount} movie(s), which is not ` +
                  `enough to find similar users.`
                : `No recommendation could be scored for User ${userId}: none of the ` +
                  'movies they have not seen were rated by a sufficiently similar user.'));
        return;
    }

    // Name the most influential neighbour to make the reasoning concrete.
    const topNeighbour = neighbours[0];
    why.textContent =
        `Because you are similar to other users (most similar: User ${topNeighbour.userId}, ` +
        `similarity ${topNeighbour.similarity.toFixed(3)}), we recommend:`;

    list.replaceChildren(...recommendations.map((rec, index) =>
        buildListItem(index + 1, rec.title, rec.score.toFixed(2))));
}

// --------------------------------------------------------------------------
// renderItemBased(userId, result, ratedCount)
// --------------------------------------------------------------------------
function renderItemBased(userId, result, ratedCount) {
    const panel = document.getElementById('item-based-result');
    const why = panel.querySelector('.panel-why');
    const list = panel.querySelector('.rec-list');

    const { recommendations, seedMovie } = result;

    if (recommendations.length === 0) {
        why.textContent = 'Because you liked certain movies, we recommend:';
        list.replaceChildren(buildEmptyMessage(
            ratedCount < MIN_CO_RATED
                ? `User ${userId} has rated only ${ratedCount} movie(s), which is not ` +
                  'enough to find similar movies.'
                : `No recommendation could be scored for User ${userId}: none of the ` +
                  'unseen movies share a positive similarity with the ones already rated.'));
        return;
    }

    why.textContent = seedMovie
        ? `Because you liked "${seedMovie.title}" (your rating: ` +
          `${seedMovie.rating}/5), we recommend:`
        : 'Because you liked certain movies, we recommend:';

    list.replaceChildren(...recommendations.map((rec, index) =>
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
