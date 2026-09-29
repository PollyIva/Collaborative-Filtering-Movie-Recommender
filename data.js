// ============================================================================
// data.js — Data loading & parsing module (MovieLens 100K)
//
// Responsibilities ONLY:
//   * fetch + decode u.item and u.data
//   * parse them into the `movies` and `ratings` arrays
//   * build the dense rating matrix used by the CF code in script.js
//
// This file contains no recommendation logic and no UI rendering beyond a
// single error message, so the algorithm stays isolated in script.js.
// ============================================================================

// --- Global data containers (read by script.js) -----------------------------
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// ratingMatrix[userId][itemId] = rating in [1..5], or 0 when the user has not
// rated that movie. Shape is (numUsers + 1) x (numMovies + 1) because MovieLens
// ids are 1-based and contiguous, so the id can be used directly as an index.
let ratingMatrix = null;

// Parallel boolean mask of the same shape. ratingMatrix uses 0 for "missing",
// and this makes that explicit: rated[u][i] === true means a real rating exists.
// MovieLens only ever stores ratings 1..5, so 0 is unambiguous here, but the
// mask keeps the "missing vs rated" distinction explicit for the similarity
// function instead of relying on a magic value.
let ratedMask = null;

// Inverted indexes built alongside the matrix, because the collaborative
// filtering code needs to walk the sparse data rather than scan 1.6M cells:
//   userItems[userId] -> [[itemId, rating], ...]  movies that user rated
//   itemUsers[itemId] -> [[userId, rating], ...]  users who rated that movie
// Entries are stored as [id, rating] pairs so the CF code needs no second lookup.
let userItems = null;
let itemUsers = null;

// The 18 named genres stored in each u.item line. The file actually holds 19
// binary flags: the first one is an unnamed "unknown" genre column, so the
// genre flags live in fields 6..23 and map 1:1 onto this list.
const GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// --------------------------------------------------------------------------
// loadData()
//
// Fetches both files, parses them, then derives numUsers / numMovies and builds
// the rating matrix. Must be awaited before any recommendation code runs.
// u.item is fetched and awaited before u.data, as required.
// --------------------------------------------------------------------------
async function loadData() {
    try {
        // 1) Movie catalogue (u.item).
        // NOTE: u.item is encoded in Latin-1 (ISO-8859-1), so response.text()
        // — which assumes UTF-8 — garbles accented titles ("VZr" instead of
        // "Véronique"). We decode the raw bytes with TextDecoder instead.
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load u.item (HTTP ${moviesResponse.status})`);
        }
        const moviesText = new TextDecoder('latin1').decode(
            await moviesResponse.arrayBuffer()
        );
        parseItemData(moviesText);

        // 2) Ratings (u.data). Pure ASCII, decoded the same way for consistency.
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load u.data (HTTP ${ratingsResponse.status})`);
        }
        const ratingsText = new TextDecoder('latin1').decode(
            await ratingsResponse.arrayBuffer()
        );
        parseRatingData(ratingsText);

        // 3) Derive the dimensions of the matrix.
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = movies.length;

        // 4) Build the dense matrix that the CF code reads.
        buildRatingMatrix();

        console.log(
            `Loaded ${numMovies} movies, ${ratings.length} ratings, ${numUsers} users.`
        );
    } catch (error) {
        console.error('Error loading data:', error);
        showLoadError(error);
        // Re-throw so script.js can mark initialisation as failed.
        throw error;
    }
}

// --------------------------------------------------------------------------
// showLoadError(error)
// Surfaces a load failure in the result area. Defined here (rather than in
// script.js) because data.js owns the fetching and must report its own errors.
// --------------------------------------------------------------------------
function showLoadError(error) {
    // A file:// page cannot fetch sibling files (CORS), so that failure mode
    // gets its own hint — it is by far the most common way to run this wrong.
    const isFileProtocol = location.protocol === 'file:';
    const hint = isFileProtocol
        ? ' Browsers block fetch() on file:// URLs — serve the folder over HTTP '
          + '(e.g. "python -m http.server") or publish it to GitHub Pages.'
        : ' Make sure u.item and u.data are in the same folder as index.html.';

    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = `Error: ${error.message}.${hint}`;
        statusEl.classList.add('error');
    }

    const message = document.createElement('p');
    message.className = 'rec-empty';
    message.textContent = `Could not load the MovieLens data: ${error.message}${hint}`;
    document.getElementById('user-based-result')
        .querySelector('.rec-list')
        .replaceChildren(message);
    document.getElementById('item-based-result')
        .querySelector('.rec-list')
        .replaceChildren(message.cloneNode(true));
}

// --------------------------------------------------------------------------
// parseItemData(text)
//
// Each u.item line is pipe-separated:
//   id | title | release date | video date | url | 19 genre flags
// Field 0 = id, field 1 = title, fields 6..23 = the 18 named genre flags.
// Pushes { id, title, genres } into `movies`.
// --------------------------------------------------------------------------
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 24) continue; // need id, title and all genre flags

        const id = parseInt(fields[0], 10);
        const title = fields[1];

        // Collect the names of the genres flagged with '1'.
        const genres = [];
        for (let g = 0; g < GENRES.length; g++) {
            if (fields[6 + g] === '1') {
                genres.push(GENRES[g]);
            }
        }

        movies.push({ id, title, genres });
    }
}

// --------------------------------------------------------------------------
// parseRatingData(text)
//
// Each u.data line is tab-separated:
//   userID \t itemID \t rating \t timestamp
// Pushes { userId, itemId, rating, timestamp } into `ratings`.
// --------------------------------------------------------------------------
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue;

        ratings.push({
            userId: parseInt(fields[0], 10),
            itemId: parseInt(fields[1], 10),
            rating: parseInt(fields[2], 10),
            timestamp: parseInt(fields[3], 10)
        });
    }
}

// --------------------------------------------------------------------------
// buildRatingMatrix()
//
// Builds the structures that the collaborative-filtering code works with:
//
//   ratingMatrix[u][i] -> the rating the user gave the movie, or 0 if missing
//   ratedMask[u][i]    -> true when a real rating exists
//   userItems / itemUsers -> the same data as sparse inverted indexes
//
// A typed array is used for the matrix because it holds ~1.6M cells; a plain
// array of arrays is both slower and considerably heavier on memory.
// --------------------------------------------------------------------------
function buildRatingMatrix() {
    const rows = numUsers + 1;
    const cols = numMovies + 1;

    ratingMatrix = Array.from({ length: rows }, () => new Float32Array(cols));
    ratedMask = Array.from({ length: rows }, () => new Uint8Array(cols));
    userItems = Array.from({ length: rows }, () => []);
    itemUsers = Array.from({ length: cols }, () => []);

    for (const { userId, itemId, rating } of ratings) {
        ratingMatrix[userId][itemId] = rating;
        ratedMask[userId][itemId] = 1;
        userItems[userId].push([itemId, rating]);
        itemUsers[itemId].push([userId, rating]);
    }
}
