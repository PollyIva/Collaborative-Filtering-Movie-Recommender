// ============================================================================
// results.js — the report's §4 experiments, rendered on the page.
//
// This is a companion view of §4 "Experiments" in
// Polina_Ivanilova_a03_report.md. Every number below is the number printed in
// the report — nothing here is a separate claim.
//
//   Tables 1-3  the worked example, transcribed from the report
//   Table 4     pair space            <- experiments/results.json (exact counts)
//   Table 5     user 1 Top-5          <- computed live by this page
//   Tables 6-7  agreement and cost    <- experiments/results.json / report
//
// Re-run `python experiments/run.py` to regenerate results.json.
// ============================================================================

const RESULTS_URL = 'experiments/results.json';

// --- the worked example, as printed in report Tables 1-3 ---------------------
const WORKED = {
    // Report Table 1 — interaction matrix, five users against twelve films.
    matrix: {
        cols: ['19', '48', '174', '178', '22', '187', '238', '50', '181', '286', '1', '13', 'rated'],
        rows: [
            ['u1',   [5, 5, 5, 5, 4, 4, 4, 5, 5, 0, 5, 5], 272],
            ['u876', [5, 5, 4, 4, 4, 4, 4, 0, 0, 5, 0, 0], 21],
            ['u516', [0, 0, 0, 0, 0, 0, 0, 5, 4, 5, 0, 0], 21],
            ['u895', [0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 4, 5], 20],
            ['u105', [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0], 23],
        ],
    },
    // Report Table 2 — user-user cosine, co-rated count in brackets.
    userCos: {
        users: ['u1', 'u876', 'u516', 'u895', 'u105'],
        rated: [272, 21, 21, 20, 23],
        values: [
            ['1.000', '0.9949 (7)', '0.9926 (10)', '0.9886 (8)', '0.9877 (7)'],
            ['0.9949 (7)', '1.000', 'n/a', 'n/a', 'n/a'],
            ['0.9926 (10)', 'n/a', '1.000', 'n/a', 'n/a'],
            ['0.9886 (8)', 'n/a', 'n/a', '1.000', 'n/a'],
            ['0.9877 (7)', 'n/a', 'n/a', 'n/a', '1.000'],
        ],
    },
    // Report Table 3 — item-item cosine around the seed *Hoop Dreams*.
    itemCos: {
        items: ['48', '372', '1214', '645', '115'],
        titles: ['Hoop Dreams', 'Jeffrey', 'In the Realm of the Senses',
                 'Paris Is Burning', 'Haunted World of Edward D. Wood Jr., The'],
        genres: ['Documentary', 'Comedy', 'Drama', 'Documentary', 'Documentary'],
        raters: [117, 7, 5, 11, 7],
        values: [
            ['1.000', '0.9962', '0.9935', '0.9934', '0.9923'],
            ['0.9962', '1.000', 'n/a', '0.9615', 'n/a'],
            ['0.9935', 'n/a', '1.000', 'n/a', 'n/a'],
            ['0.9934', '0.9615', 'n/a', '1.000', 'n/a'],
            ['0.9923', 'n/a', 'n/a', 'n/a', '1.000'],
        ],
    },
};

// Report Table 7 — timings from the report's own run. Timings are
// machine-dependent, so these are pinned to the report rather than re-measured.
const REPORTED_COST = [
    ['Item-item precompute (one-off, chunked)', '313 ms'],
    ['User-Based query', '5.59 ms'],
    ['Item-Based query', '1.54 ms'],
    ['Page ready to interact', '≈1.0 s local / 2.1 s from GitHub Pages'],
];

// --- formatting helpers -----------------------------------------------------
const group = n => n.toLocaleString('en-US');
const pct = (part, whole) => `${(100 * part / whole).toFixed(1)}%`;
const fixed = (n, d) => n.toFixed(d);

// --- table builder ----------------------------------------------------------
// `head` = column titles; `rows` = arrays of pre-formatted strings.
// A leading '' marks the row-label column. `diag` bolds the diagonal.
function buildTable(head, rows, opts) {
    const o = opts || {};
    const table = document.createElement('table');
    table.className = 'data-table';

    if (o.caption) {
        const cap = document.createElement('caption');
        cap.textContent = o.caption;
        table.appendChild(cap);
    }

    const thead = document.createElement('thead');
    const htr = document.createElement('tr');
    head.forEach((h, i) => {
        const th = document.createElement('th');
        th.textContent = h;
        if (i === 0) th.className = 'corner';
        htr.appendChild(th);
    });
    thead.appendChild(htr);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach((row, r) => {
        const tr = document.createElement('tr');
        row.forEach((cell, i) => {
            const td = document.createElement(i === 0 ? 'th' : 'td');
            if (i === 0) td.className = 'row-label';
            if (o.diag && i === r + 1) td.className = 'diag';
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

function block(title, source) {
    const b = document.createElement('div');
    b.className = 'exp-block';
    const h = document.createElement('h3');
    h.textContent = title;
    b.appendChild(h);
    if (source) {
        const s = document.createElement('p');
        s.className = 'exp-source';
        s.textContent = source;
        b.appendChild(s);
    }
    return b;
}

function note(html) {
    const p = document.createElement('p');
    p.className = 'exp-note';
    p.innerHTML = html;
    return p;
}

function heading(text) {
    const h = document.createElement('h4');
    h.className = 'exp-h4';
    h.textContent = text;
    return h;
}

// --- render -----------------------------------------------------------------
function renderWorkedExample(root) {
    const W = WORKED;

    // Report Table 1
    const b1 = block('Worked example', 'Report Tables 1–3 · user 1 (272 ratings) and its four nearest neighbours');
    b1.appendChild(heading('Table 1 · interaction matrix, five users against twelve films (0 = unrated)'));
    b1.appendChild(buildTable(
        ['user \\ item', ...W.matrix.cols],
        W.matrix.rows.map(([u, vals, rated]) => [u, ...vals.map(String), String(rated)]),
        { diag: false }
    ));
    b1.appendChild(note(
        `Columns 19–238 are the seven films user 1 and user 876 share &mdash; the entire basis for the
         neighbour score <code>cos = 138.0 / (12.17 &times; 11.40) = 0.9949</code>, all four- or five-star.`));
    root.appendChild(b1);

    // Report Table 2
    const b2 = block(null, 'Report Table 2 · user-user cosine, co-rated count in brackets');
    b2.appendChild(buildTable(
        ['', ...W.userCos.users, 'rated'],
        W.userCos.values.map((row, r) => [W.userCos.users[r], ...row, String(W.userCos.rated[r])]),
        { diag: true }
    ));
    b2.appendChild(note(
        `<code>n/a</code> = fewer than five co-raters. Every off-diagonal cell below the first row is
         <code>n/a</code>: the four neighbours share only 1&ndash;2 films <em>with each other</em> while
         sharing 7&ndash;10 with the target.`));
    root.appendChild(b2);

    // Report Table 3
    const b3 = block(null, 'Report Table 3 · item-item cosine around the dropdown seed, Hoop Dreams (48, Documentary, 117 raters)');
    b3.appendChild(buildTable(
        ['item \\ item', ...W.itemCos.items],
        W.itemCos.values.map((row, r) => [W.itemCos.items[r], ...row]),
        { diag: true }
    ));
    b3.appendChild(buildTable(
        ['item', 'film', 'genre', 'raters'],
        W.itemCos.items.map((id, i) => [
            id, W.itemCos.titles[i], W.itemCos.genres[i], String(W.itemCos.raters[i]),
        ]),
        { caption: 'The columns, and the failure they reveal' }
    ));
    b3.appendChild(note(
        `With no seed the app takes the highest-rated film, <em>Toy Story</em>. A <strong>Comedy
         outranks a Documentary</strong> &mdash; the seven users behind <em>Jeffrey</em> rated the pair
         5/4, 5/4, 4/3, 4/3, 4/3, 3/3, 1/1, so genre loses to sample size.`));
    root.appendChild(b3);
}

function renderPairSpace(root, R) {
    const ps = R.pairSpace;
    const b = block('Results (A) · pair space',
                    'Report Table 4 · exact counts over every pair, not a sample');
    b.appendChild(buildTable(
        ['', 'User-Based', 'Item-Based'],
        [
            ['Entities', group(R.dataset.users), group(R.dataset.movies)],
            ['Distinct pairs', group(ps.user.pairs), group(ps.item.pairs)],
            ['Pairs with zero overlap',
                `${group(ps.user.zero)} (${pct(ps.user.zero, ps.user.pairs)})`,
                `${group(ps.item.zero)} (${pct(ps.item.zero, ps.item.pairs)})`],
            ['Pairs with 1–4 co-rated',
                `${group(ps.user.low)} (${pct(ps.user.low, ps.user.pairs)})`,
                `${group(ps.item.low)} (${pct(ps.item.low, ps.item.pairs)})`],
            ['Pairs clearing MIN_CO_RATED=5',
                `${group(ps.user.clearing)} (${pct(ps.user.clearing, ps.user.pairs)})`,
                `${group(ps.item.clearing)} (${pct(ps.item.clearing, ps.item.pairs)})`],
        ]
    ));
    b.appendChild(note(
        `Users are dense and items are not: most item pairs never reach five shared ratings, so the
         floor discards most of that axis before any similarity is computed, and
         ${group(ps.item.zero)} pairs with an undefined cosine vanish uncounted.`));
    root.appendChild(b);
}

function renderAgreement(root, R) {
    const ag = R.agreement;
    const b = block('Results (B) · saturation and agreement',
                    'Report Tables 5–6 · all 943 users, live build, no explicit seed');
    root.appendChild(b);

    // Report Table 5 — live
    const slot = document.createElement('div');
    slot.innerHTML = '<p class="exp-note">Table 5 is recomputed live in your browser for user 1.</p>';
    b.appendChild(slot);
    whenReady(() => {
        const ub = getUserBasedRecommendations(1, 5);
        const ib = getItemBasedRecommendations(1, 5, 0);
        const rows = [];
        for (let i = 0; i < 5; i++) {
            rows.push([
                String(i + 1),
                ub[i] ? `${ub[i].title} — ${ub[i].score.toFixed(2)}` : '—',
                ib[i] ? `${ib[i].title} — ${ib[i].score.toFixed(2)}` : '—',
            ]);
        }
        slot.innerHTML = '';
        slot.appendChild(heading('Table 5 · Top-5, user 1, read back from the running page'));
        slot.appendChild(buildTable(['Rank', 'User-Based', 'Item-Based'], rows));
    });

    // Report Table 6
    b.appendChild(heading('Table 6 · saturation and agreement over all 943 users'));
    b.appendChild(buildTable(
        ['', 'User-Based', 'Item-Based'],
        [
            ['Top-1 exactly 5.0',
                `${ag.userBased.top1Exactly5} users (${fixed(ag.userBased.top1Exactly5Pct, 1)}%)`,
                `${ag.itemBased.top1Exactly5} users (${fixed(ag.itemBased.top1Exactly5Pct, 1)}%)`],
            ['All five Top-5 exactly 5.0',
                `${ag.userBased.all5Exactly5} users (${fixed(ag.userBased.all5Exactly5Pct, 1)}%)`,
                `${ag.itemBased.all5Exactly5} users (${fixed(ag.itemBased.all5Exactly5Pct, 1)}%)`],
            ['Mean Top-1', fixed(ag.userBased.meanTop1, 3), fixed(ag.itemBased.meanTop1, 3)],
        ]
    ));
    b.appendChild(note(
        `Both methods return five identical-looking scores &mdash; Item-Based for
         ${fixed(ag.itemBased.all5Exactly5Pct, 1)}% of users &mdash; and a User-Based mean of
         ${fixed(ag.userBased.meanTop1, 3)} means its Top-1 merely <em>displays</em> as
         <code>5.00</code>. The lists share ${ag.sharedTitles} titles: mean overlap
         ${fixed(ag.meanSharedPerUser, 4)} against ${fixed(ag.expectedSharedPerUser, 4)} expected by
         chance, and ${fixed(ag.usersWithZeroOverlapPct, 1)}% of users get two completely disjoint
         lists.`));
}

function renderCost(root) {
    const b = block('Results (C) · cost',
                    'Report Table 7 · headless Chrome against the local page, medians over 40 users');
    b.appendChild(buildTable(['Stage', 'Cost'], REPORTED_COST));
    b.appendChild(note(
        `The item precompute grows with the square of the entity count and is paid once, while a user
         query re-scores 20 neighbours for every candidate on every click. Timings are
         machine-dependent; the figures above are the ones printed in the report.`));
    root.appendChild(b);
}

// --- helpers ----------------------------------------------------------------
function whenReady(fn) {
    if (typeof isReady !== 'undefined' && isReady) { fn(); return; }
    let tries = 0;
    const poll = setInterval(() => {
        if (typeof isReady !== 'undefined' && isReady) { clearInterval(poll); fn(); }
        else if (++tries > 600) {
            clearInterval(poll);
            document.querySelectorAll('#exp-tables .exp-note')
                .forEach(n => { n.textContent = 'The app did not finish loading, so Table 5 could not be computed live.'; });
        }
    }, 250);
}

async function load() {
    const root = document.getElementById('exp-tables');
    try {
        const res = await fetch(RESULTS_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const R = await res.json();
        root.classList.remove('exp-loading');
        root.innerHTML = '';
        renderWorkedExample(root);
        renderPairSpace(root, R);
        renderAgreement(root, R);
        renderCost(root);
    } catch (err) {
        root.classList.remove('exp-loading');
        root.innerHTML =
            `<p class="exp-note">Could not load <code>${RESULTS_URL}</code> (${err.message}). Serve the
             folder over HTTP rather than opening <code>index.html</code> from disk, or re-run
             <code>python experiments/run.py</code> to regenerate it.</p>`;
    }
}

document.addEventListener('DOMContentLoaded', load);
