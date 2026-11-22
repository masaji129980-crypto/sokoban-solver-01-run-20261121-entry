// Push-optimal Sokoban solver for sokoban-solver-01-run-20261121 (MIT).
//
// The levels this competition draws are a hallway with independent rooms off
// it (RULES.txt: a box pushed out of a room can never come back, so P* is the
// sum of the rooms' optima). This solver therefore splits each level into its
// rooms and solves every room push-optimally on its own, then stitches the
// player's walks between the rooms. A room holds at most 4 boxes, so a
// push-space A* with an admissible Hungarian-assignment heuristic, dead-square
// pruning and frozen 2x2-block pruning finds each room's optimum quickly.
//
// Determinism: every budget is counted in work (node expansions), never in
// wall-clock time, so two runs write the same bytes. No clock, no randomness.
//
// Run: node main.ts  (MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR override
// /input and /output, as the kit's starter does.)
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four moves: [dx, dy, step letter, push letter].
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budgets (node expansions), never the clock. Each instance gets its own
// budget, so one hard instance cannot starve the rest; the whole set's worst
// case stays far below the 60 s wall (a set of these levels solves in a few
// thousand expansions; the caps are 30-60x that headroom).
const ROOM_EXPANSIONS = 60_000;
const GREEDY_EXPANSIONS = 30_000;
const INSTANCE_EXPANSIONS = 100_000;
const WHOLE_EXPANSIONS = 60_000;
let budgetLeft = INSTANCE_EXPANSIONS;

// ---------------------------------------------------------------- parsing
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

// ------------------------------------------------------------ static tables
// Squares from which a box alone can be pushed onto some goal (pull the goals
// back over open floor). A box on any other square is dead: no push saves it.
function liveSquares(level) {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Walking distances from every square to each goal, over walls alone (a lower
// bound on the pushes a box needs to reach that goal).
function goalDistances(level) {
  const { w, wall, goals } = level;
  const dist = [];
  for (const g of goals) {
    const d = new Int32Array(wall.length).fill(-1);
    const queue = [g];
    d[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi];
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && d[q] < 0) {
          d[q] = d[p] + 1;
          queue.push(q);
        }
      }
    }
    dist.push(d);
  }
  return dist;
}

// The squares the player reaches from `from` around walls and `occupied`
// boxes, marked in `seen` with `stamp`; returns the region's lowest square
// (the region's name). The start square counts even when it is marked wall.
function reach(level, occupied, from, seen, stamp, stack) {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and
// boxes with a box off its goal: no push ever moves it again.
function frozenBlock(level, occupied, goal, t) {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// The shortest walk from `from` to `to` around walls and boxes, in lower-case
// LURD letters (first direction first on ties).
function walkTo(level, occupied, from, to) {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// ------------------------------------------------------- room decomposition
// Cut the dead squares of the player's row (the hallway: a box there reaches
// no goal) out of the floor; each part left that holds a box is a room.
// Null when the level is no set of rooms.
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// ------------------------------------------------------------- push-space A*
// All permutations of 0..n-1, for the Hungarian matching over n <= 4 boxes.
function permutations(n) {
  const out = [];
  const a = [...Array(n).keys()];
  const gen = (k) => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i], a[k]];
      gen(k + 1);
      [a[k], a[i]] = [a[i], a[k]];
    }
  };
  gen(0);
  return out;
}
const PERMS = { 1: permutations(1), 2: permutations(2), 3: permutations(3), 4: permutations(4) };

// Admissible estimate: the cheapest assignment of boxes to goals by walking
// distance (each push moves a box one square, so pushes >= walking distance).
// With more than 4 boxes, the sum of each box's nearest goal (still a lower
// bound on any assignment).
function heuristic(boxes, dist, perms) {
  if (perms) {
    let best = Infinity;
    for (const p of perms) {
      let s = 0;
      for (let i = 0; i < boxes.length; i++) {
        const d = dist[p[i]][boxes[i]];
        if (d < 0) {
          s = Infinity;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  }
  let s = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (let g = 0; g < dist.length; g++) if (dist[g][b] >= 0 && dist[g][b] < m) m = dist[g][b];
    if (m === Infinity) return Infinity;
    s += m;
  }
  return s;
}

// Push-optimal A* over one (sub-)level: {pushes, moves} or null when the
// expansion budget runs out. `greedy` runs best-first on h alone (finds any
// solution fast, not necessarily optimal) as a fallback.
function solvePushAStar(level, cap, greedy) {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dist = goalDistances(level);
  const perms = PERMS[n] ?? null;
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes) => boxes.every((b) => goal[b]);

  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  // State key: the region's lowest square and the boxes' squares. Exact as a
  // number while size ** (n + 1) < 2 ** 53 (rooms of up to 5 boxes); a string
  // otherwise.
  const numeric = Math.pow(size, n + 1) < 2 ** 53;
  const keyOf = (boxes, low) => {
    if (numeric) {
      let k = low;
      for (const b of boxes) k = k * size + b;
      return k;
    }
    return `${low}|${boxes.join(',')}`;
  };

  for (const b of level.boxes) occupied[b] = 1;
  const low0 = reach(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  const h0 = heuristic(level.boxes, dist, perms);
  if (h0 === Infinity) return null;

  const boxesArr = [level.boxes.slice()];
  const playerArr = [level.player];
  const lowArr = [low0];
  const gArr = [0];
  const fArr = [h0];
  const parentArr = [-1];
  const pushArr = [null]; // [from, to, dirIndex] that made this state
  const best = new Map();
  best.set(keyOf(level.boxes, low0), 0);

  // Binary heap of state indices on (f, -g, insertion order): deterministic.
  const heap = [0];
  let seq = 0;
  const seqArr = [0];
  const less = (a, b) => {
    if (fArr[a] !== fArr[b]) return fArr[a] < fArr[b];
    if (gArr[a] !== gArr[b]) return gArr[a] > gArr[b];
    return seqArr[a] < seqArr[b];
  };
  const heapPush = (idx) => {
    let i = heap.length;
    heap.push(idx);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    if (expanded >= cap || budgetLeft <= 0) return null;
    const s = heapPop();
    if (best.get(keyOf(boxesArr[s], lowArr[s])) !== s) continue;
    // Mark this state's player region (for the push squares' reachability).
    const boxes = boxesArr[s];
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, playerArr[s], seen, ++stamp, stack);
    const here = stamp;
    expanded++;
    budgetLeft--;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di];
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(level, occupied, goal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        const low = reach(level, occupied, b, mark, ++stamp, stack);
        const k = keyOf(next, low);
        const prevIdx = best.get(k);
        const ng = gArr[s] + 1;
        if (prevIdx !== undefined && gArr[prevIdx] <= ng) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const h = heuristic(next, dist, perms);
        occupied[t] = 0;
        occupied[b] = 1;
        if (h === Infinity) continue;
        const idx = boxesArr.length;
        boxesArr.push(next);
        playerArr.push(b);
        lowArr.push(low);
        gArr.push(ng);
        fArr.push(greedy ? h : ng + h);
        parentArr.push(s);
        pushArr.push([b, t, di]);
        seqArr.push(++seq);
        best.set(k, idx);
        heapPush(idx);
        if (solved(next)) {
          found = idx;
          break;
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = parentArr[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = boxesArr[s];
    const [from, to, di] = pushArr[s];
    const d = to - from;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[di][3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// ------------------------------------------------------------------- solving
function solveLevel(text) {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushAStar(level, Math.min(WHOLE_EXPANSIONS, budgetLeft), false)
      ?? solvePushAStar(level, Math.min(GREEDY_EXPANSIONS, budgetLeft), true);
    return whole ? whole.moves : '';
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    // The sub-level: this room's boxes and goals, the other rooms walled off
    // (an undone room wholly; a done room's goals, so the player can walk out
    // of the room it stands in). A box never needs to leave its room.
    const wall = level.wall.slice();
    const playerRoom = rooms.find((r) => r !== room && r.squares.includes(player));
    for (const other of rooms) {
      if (other === room) continue;
      if (other === playerRoom) {
        for (const g of other.goals) wall[g] = 1;
      } else {
        for (const sq of other.squares) wall[sq] = 1;
      }
    }
    const sub = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = solvePushAStar(sub, Math.min(ROOM_EXPANSIONS, budgetLeft), false);
    if (solved === null && budgetLeft > 0) {
      solved = solvePushAStar(sub, Math.min(GREEDY_EXPANSIONS, budgetLeft), true);
    }
    if (solved === null) return moves; // unsolved room: answer stays partial
    moves += solved.moves;
    // The player ends on the last pushed box's old square.
    let p = player;
    for (const ch of solved.moves.toLowerCase()) p += ch === 'u' ? -level.w : ch === 'd' ? level.w : ch === 'l' ? -1 : 1;
    player = p;
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let dir;
  try {
    dir = statSync(join(I, id));
  } catch {
    continue;
  }
  if (!dir.isDirectory()) continue;
  let moves = '';
  budgetLeft = INSTANCE_EXPANSIONS; // each instance gets a fresh work budget
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
