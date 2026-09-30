// ============================================================================
// experiments/measure.js
//
// Every number quoted in section 4 of the report is produced by this file,
// against the *shipped* data.js + script.js running in a real browser. There
// is no reimplementation of the CF code here: the harness mutates the live
// globals (ratingMatrix / ratedMask / userItems / itemUsers), rebuilds the real
// itemSim cache, and calls the real getUserBasedRecommendations /
// getItemBasedRecommendations.
//
// Injected into the running page by experiments/run.py. Not referenced by
// index.html, so the app itself never loads it.
// ============================================================================

window.runMeasurements = async function (cfg) {
    const K = cfg.k || 10;                    // NDCG cut-off
    const REL = cfg.relThreshold || 4;        // a held-out rating >= REL is relevant
    const N_EVAL = cfg.users || 250;          // held-out evaluation sample
    const HOLDOUT = cfg.holdoutFrac || 0.2;   // fraction of each user's ratings hidden
    const SEED = cfg.seed >>> 0;              // fixed so every run is reproducible
    const COST_USERS = cfg.costUsers || 40;   // users timed for query cost
    const log = msg => console.log('[measure] ' + msg);

    const U = numUsers;
    const I = numMovies;
    const out = { config: { K, REL, N_EVAL, HOLDOUT, SEED, COST_USERS } };

    // --- helpers ------------------------------------------------------------
    const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
    const median = a => {
        const s = a.slice().sort((x, y) => x - y);
        const n = s.length;
        return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    };

    let rngState = SEED;
    const rnd = () => {                        // mulberry32
        rngState = rngState + 0x6D2B79F5 | 0;
        let t = Math.imul(rngState ^ rngState >>> 15, 1 | rngState);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    const shuffle = arr => {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(rnd() * (i + 1));
            const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
        return arr;
    };

    // --- 0. make a returned title identify its item id ---------------------
    // The brief fixes the return type at { title, score }, so the harness
    // prefixes the id. Titles never enter a score or a sort, so this cannot
    // change a single ranking — it only makes titles uniquely invertible.
    const realTitle = getMovieTitle;
    const SEP = '\u0000';
    getMovieTitle = function (itemId) { return itemId + SEP + realTitle(itemId); };
    const idOf = rec => parseInt(rec.title.split(SEP)[0], 10);

    // --- 1. dataset facts ---------------------------------------------------
    log('dataset');
    const perUser = [];
    const perItem = [];
    for (let u = 1; u <= U; u++) perUser.push(userItems[u].length);
    for (let i = 1; i <= I; i++) perItem.push(itemUsers[i].length);

    const cells = (U + 1) * (I + 1);          // the allocated (numUsers+1)x(numMovies+1) grid
    out.dataset = {
        users: U,
        movies: I,
        ratings: ratings.length,
        cells: cells,
        emptyPct: 100 * (cells - ratings.length) / cells,
        perUser: { min: Math.min(...perUser), median: median(perUser), max: Math.max(...perUser) },
        perItem: {
            min: Math.min(...perItem),
            median: median(perItem),
            max: Math.max(...perItem),
            under5: perItem.filter(n => n < 5).length
        }
    };

    // --- 2. pair space, exact over every pair ------------------------------
    // Co-rating counts are accumulated the same way buildItemSimilarities
    // accumulates its own: walk the inverted index and count each pair once.
    log('pair space');
    const uw = U + 1, iw = I + 1;
    const userCo = new Int32Array(uw * uw);
    for (let i = 1; i <= I; i++) {
        const us = itemUsers[i];
        for (let a = 0; a < us.length; a++) {
            const ua = us[a][0];
            for (let b = a + 1; b < us.length; b++) {
                const ub = us[b][0];
                userCo[(ua < ub ? ua : ub) * uw + (ua < ub ? ub : ua)]++;
            }
        }
    }
    const itemCo = new Int32Array(iw * iw);
    for (let u = 1; u <= U; u++) {
        const its = userItems[u];
        for (let a = 0; a < its.length; a++) {
            const ia = its[a][0];
            for (let b = a + 1; b < its.length; b++) {
                const ib = its[b][0];
                itemCo[(ia < ib ? ia : ib) * iw + (ia < ib ? ib : ia)]++;
            }
        }
    }

    const bucket = (co, w, n) => {
        let zero = 0, low = 0, high = 0;
        for (let i = 1; i <= n; i++) {
            for (let j = i + 1; j <= n; j++) {
                const c = co[i * w + j];
                if (c === 0) zero++; else if (c < 5) low++; else high++;
            }
        }
        const total = n * (n - 1) / 2;
        return { pairs: total, zero, low, high, clearing: high };
    };
    out.pairSpace = {
        user: bucket(userCo, uw, U),
        item: bucket(itemCo, iw, I)
    };

    // --- 3. agreement, all users on the full matrix -------------------------
    log('agreement over ' + U + ' users');
    const ag = {
        ub: { n: 0, top1Exactly5: 0, all5Exactly5: 0, top1Sum: 0 },
        ib: { n: 0, top1Exactly5: 0, all5Exactly5: 0, top1Sum: 0 },
        sharedTitles: 0, compared: 0, zeroOverlap: 0
    };
    for (let u = 1; u <= U; u++) {
        const ub = getUserBasedRecommendations(u, 5);
        const ib = getItemBasedRecommendations(u, 5, 0);   // 0 = no explicit seed
        if (ub.length) {
            ag.ub.n++;
            ag.ub.top1Sum += ub[0].score;
            if (ub[0].score === 5) ag.ub.top1Exactly5++;
            if (ub.every(r => r.score === 5)) ag.ub.all5Exactly5++;
        }
        if (ib.length) {
            ag.ib.n++;
            ag.ib.top1Sum += ib[0].score;
            if (ib[0].score === 5) ag.ib.top1Exactly5++;
            if (ib.every(r => r.score === 5)) ag.ib.all5Exactly5++;
        }
        if (ub.length && ib.length) {
            const s = new Set(ub.map(idOf));
            let shared = 0;
            for (const r of ib) if (s.has(idOf(r))) shared++;
            ag.compared++;
            ag.sharedTitles += shared;
            if (shared === 0) ag.zeroOverlap++;
        }
    }
    const fin = s => ({
        usersWithAList: s.n,
        top1Exactly5: s.top1Exactly5,
        top1Exactly5Pct: 100 * s.top1Exactly5 / s.n,
        all5Exactly5: s.all5Exactly5,
        all5Exactly5Pct: 100 * s.all5Exactly5 / s.n,
        meanTop1: s.top1Sum / s.n
    });
    out.agreement = {
        userBased: fin(ag.ub),
        itemBased: fin(ag.ib),
        compared: ag.compared,
        sharedTitles: ag.sharedTitles,
        usersWithZeroOverlap: ag.zeroOverlap,
        usersWithZeroOverlapPct: 100 * ag.zeroOverlap / ag.compared,
        meanSharedPerUser: ag.sharedTitles / ag.compared,
        expectedSharedPerUser: 25 / I
    };

    // --- 4. cost ------------------------------------------------------------
    log('cost');
    const t0 = performance.now();
    await buildItemSimilarities();
    const precomputeMs = performance.now() - t0;

    for (let u = 1; u <= 20; u++) {            // warm up the JIT before timing
        getUserBasedRecommendations(u, 5);
        getItemBasedRecommendations(u, 5, 0);
    }
    const ubMs = [], ibMs = [];
    for (let u = 1; u <= COST_USERS; u++) {
        let t = performance.now();
        getUserBasedRecommendations(u, 5);
        ubMs.push(performance.now() - t);
        t = performance.now();
        getItemBasedRecommendations(u, 5, 0);
        ibMs.push(performance.now() - t);
    }
    out.cost = {
        precomputeMs: precomputeMs,
        userQueryMsMedian: median(ubMs),
        itemQueryMsMedian: median(ibMs),
        timedUsers: COST_USERS
    };

    // --- 5. held-out ranking quality, NDCG@K --------------------------------
    log('held-out NDCG@' + K + ' over ' + N_EVAL + ' users');
    const pristine = {
        matrix: ratingMatrix.map(r => r.slice()),
        mask: ratedMask.map(r => r.slice()),
        userItems: userItems.map(a => a.slice()),
        itemUsers: itemUsers.map(a => a.slice())
    };
    const restoreAll = () => {
        for (let u = 1; u <= U; u++) {
            ratingMatrix[u].set(pristine.matrix[u]);
            ratedMask[u].set(pristine.mask[u]);
        }
        for (let u = 0; u <= U; u++) userItems[u] = pristine.userItems[u].slice();
        for (let i = 0; i <= I; i++) itemUsers[i] = pristine.itemUsers[i].slice();
    };

    const pool = [];
    for (let u = 1; u <= U; u++) if (pristine.userItems[u].length >= 20) pool.push(u);
    const evalUsers = shuffle(pool.slice()).sort((a, b) => a - b).slice(0, N_EVAL);

    let idcgZero = 0, heldTotal = 0, heldRelTotal = 0;
    const acc = { userBased: [], itemBased: [], itemBasedUnguarded: [], random: [] };
    let empty = { userBased: 0, itemBased: 0, itemBasedUnguarded: 0 };

    const ndcg = (recs, heldRatings) => {
        let rel = 0;
        for (const r of heldRatings.values()) if (r >= REL) rel++;
        const ideal = Math.min(K, rel);
        if (ideal === 0) { idcgZero++; return 0; }
        let idcg = 0;
        for (let r = 0; r < ideal; r++) idcg += 1 / Math.log2(r + 2);
        let dcg = 0;
        for (let r = 0; r < Math.min(K, recs.length); r++) {
            const rating = heldRatings.get(idOf(recs[r]));
            if (rating !== undefined && rating >= REL) dcg += 1 / Math.log2(r + 2);
        }
        return dcg / idcg;
    };

    for (let idx = 0; idx < evalUsers.length; idx++) {
        const u = evalUsers[idx];
        if (idx % 20 === 0) log('  split ' + (idx + 1) + '/' + evalUsers.length);

        // --- hide a random HOLDOUT fraction of this user's ratings ----------
        const all = pristine.userItems[u].map(e => e[0]);
        const held = shuffle(all.slice()).slice(0, Math.max(1, Math.round(all.length * HOLDOUT)));
        const heldSet = new Set(held);
        const heldRatings = new Map();
        for (const [i, r] of pristine.userItems[u]) if (heldSet.has(i)) heldRatings.set(i, r);
        heldTotal += held.length;
        heldRelTotal += [...heldRatings.values()].filter(r => r >= REL).length;

        for (const i of held) { ratingMatrix[u][i] = 0; ratedMask[u][i] = 0; }
        userItems[u] = pristine.userItems[u].filter(e => !heldSet.has(e[0]));
        for (const i of held) itemUsers[i] = pristine.itemUsers[i].filter(e => e[0] !== u);

        // guarded cache -> the two shipped recommenders
        await buildItemSimilarities();
        const ub = getUserBasedRecommendations(u, K);
        const ib = getItemBasedRecommendations(u, K, 0);
        if (!ub.length) empty.userBased++;
        if (!ib.length) empty.itemBased++;

        // unguarded cache -> the ablation, same code with the floor removed
        await buildItemSimilarities(0);
        const ibU = getItemBasedRecommendations(u, K, 0);
        if (!ibU.length) empty.itemBasedUnguarded++;

        // random baseline drawn from exactly the pool the recommenders see:
        // every item this user has not rated in the training matrix
        const cand = [];
        for (let i = 1; i <= I; i++) if (!ratedMask[u][i]) cand.push(i);
        const rand = shuffle(cand).slice(0, K).map(i => ({ title: i + SEP + realTitle(i) }));

        acc.userBased.push(ndcg(ub, heldRatings));
        acc.itemBased.push(ndcg(ib, heldRatings));
        acc.itemBasedUnguarded.push(ndcg(ibU, heldRatings));
        acc.random.push(ndcg(rand, heldRatings));

        restoreAll();
    }

    await buildItemSimilarities();            // leave the page in the app's normal state

    out.ndcg = {
        users: evalUsers.length,
        meanHeldOut: heldTotal / evalUsers.length,
        meanRelevantHeldOut: heldRelTotal / evalUsers.length,
        usersWithNoRelevantHoldout: idcgZero,
        emptyLists: empty,
        mean: {
            userBased: mean(acc.userBased),
            itemBased: mean(acc.itemBased),
            itemBasedUnguarded: mean(acc.itemBasedUnguarded),
            random: mean(acc.random)
        },
        median: {
            userBased: median(acc.userBased),
            itemBased: median(acc.itemBased),
            itemBasedUnguarded: median(acc.itemBasedUnguarded),
            random: median(acc.random)
        }
    };

    getMovieTitle = realTitle;                 // undo the title patch
    log('done');
    return out;
};
