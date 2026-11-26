"""Compare greedy legal policy moves or exhaustive one-ply value choices, swapping seats.

Run a project ONNX against a project or original uttt.ai network. The two-ply
suite enumerates every legal ordered start. The six-ply suite samples different
rotation/reflection families. Each network always uses one orientation. Neither
mode runs MCTS or an endgame solver.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .distill_oracle import children
from .suite_exclusion import _transforms


def initial_state():
    state = np.zeros(280, dtype=np.uint8)
    state[162:243] = 1
    state[279] = 1
    return state


def openings(depth, count, seed):
    first = initial_state()
    if depth == 2:
        starts = []
        for move in range(81):
            child, _, _ = children(first[None], np.array([0]), np.array([move]))
            starts.extend((move, int(reply)) for reply in np.flatnonzero(child[0, 162:243]))
        if len(starts) != 720 or len(set(starts)) != 720:
            raise ValueError("Invalid exhaustive two-ply suite")
        return starts
    if depth != 6 or count < 1:
        raise ValueError("Choose a six-ply sample or the complete two-ply suite")
    rng = np.random.default_rng(seed)
    maps = [(9 * p[:, None] + p[None, :]).ravel() for p in _transforms()]
    seen = set()
    starts = []
    while len(starts) < count:
        state = first[None]
        moves = []
        for _ in range(depth):
            move = int(rng.choice(np.flatnonzero(state[0, 162:243])))
            moves.append(move)
            state, _, _ = children(state, np.array([0]), np.array([move]))
        family = min(tuple(p[moves]) for p in maps)
        if family not in seen:
            seen.add(family)
            starts.append(tuple(moves))
    return starts


ORDER = np.arange(81).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).ravel()


def original_input(x):
    own = x[:, :81][:, ORDER]
    other = x[:, 81:162][:, ORDER]
    turn = 1 + 2 * (own.sum(1, keepdims=True, dtype=np.float32) - other.sum(1, keepdims=True, dtype=np.float32))
    planes = np.stack((own, other, np.broadcast_to(turn, own.shape), x[:, 162:243][:, ORDER]), axis=1)
    return planes.reshape(-1, 4, 9, 9)


class Network:
    def __init__(self, path, threads, chunk):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        self.original = inputs[0].name == 'input'
        self.heads = ('policy_logits', 'state_value') if self.original else ('policy', 'value')
        self.name = inputs[0].name
        if len(inputs) != 1 or self.name not in ('input', 'state') or set(self.heads) != {o.name for o in self.session.get_outputs()}:
            raise ValueError('Unrecognized policy/value interface')
        self.chunk = 1 if inputs[0].shape[0] == 1 else chunk
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def __call__(self, positions, values):
        out = []
        for start in range(0, len(positions), self.chunk):
            x = positions[start:start + self.chunk].astype(np.float32)
            feed = original_input(x) if self.original else x
            pred, = self.session.run([self.heads[int(values)]], {self.name: feed})
            if values:
                pred = pred.reshape(len(x))
            else:
                pred = pred.reshape(len(x), 81)
                if self.original:
                    pred = pred[:, ORDER]
            if not np.isfinite(pred).all() or (values and (np.abs(pred) > 1.00001).any()):
                raise ValueError('Invalid network predictions')
            out.append(pred)
        return np.concatenate(out)


def choose(states, network, one_ply):
    legal = states[:, 162:243].astype(bool)
    if not legal.any(1).all():
        raise ValueError('Terminal state supplied to a player')
    if not one_ply:
        logits = network(states, False)
        return np.where(legal, logits, -np.inf).argmax(1), len(states)
    row, move = np.nonzero(legal)
    child, terminal, won = children(states, row, move)
    scores = won.astype(np.float32)
    live = ~terminal
    if live.any():
        scores[live] = -network(child[live], True)
    choices = np.full(legal.shape, -np.inf, np.float32)
    choices[row, move] = scores
    return choices.argmax(1), int(live.sum())


def play(starts, networks, one_ply, parallel):
    games = []
    evaluated = [0, 0]
    for start in range(0, len(starts), parallel):
        batch = np.array(starts[start:start + parallel], dtype=np.int64)
        histories = np.repeat(batch, 2, axis=0)
        rows = np.arange(len(histories))
        states = np.tile(initial_state(), (len(histories), 1))
        for move in histories.T:
            states, terminal, _ = children(states, rows, move)
            if terminal.any():
                raise ValueError('Terminal opening')
        order = np.tile([0, 1], len(batch))
        moves = [r.tolist() for r in histories]
        winners = [None] * len(rows)
        active = rows
        for ply in range(histories.shape[1], 81):
            actors = (order[active] + ply) % 2
            chosen = np.empty(len(active), np.int64)
            for i, network in enumerate(networks):
                take = actors == i
                if take.any():
                    chosen[take], calls = choose(states[take], network, one_ply)
                    evaluated[i] += calls
            for game, move in zip(active, chosen, strict=True):
                moves[game].append(int(move))
            states, terminal, won = children(states, np.arange(len(active)), chosen)
            for game, victory in zip(active[terminal], won[terminal], strict=True):
                winners[game] = ('X' if ply % 2 == 0 else 'O') if victory else 'draw'
            active = active[~terminal]
            states = states[~terminal]
            if not len(active):
                break
        if None in winners:
            raise ValueError("Game did not terminate")
        for j, opening in enumerate(batch):
            games.append(dict(opening=opening.tolist(), firstAsX=dict(moves=moves[2*j], winner=winners[2*j]),
                              firstAsO=dict(moves=moves[2*j+1], winner=winners[2*j+1])))
    return games, evaluated


def summarize(games):
    by_seat = {}
    scores = []
    for key, seat in [('firstAsX', 'X'), ('firstAsO', 'O')]:
        points = np.array([.5 if row[key]['winner'] == 'draw' else float(row[key]['winner'] == seat) for row in games])
        by_seat[seat] = dict(wins=int((points == 1).sum()), draws=int((points == .5).sum()),
                             losses=int((points == 0).sum()), score=float(points.mean()))
        scores.append(points)
    totals = {k: sum(a[k] for a in by_seat.values()) for k in ('wins', 'draws', 'losses')}
    return dict(**totals, score=float(np.mean(scores)), bySeat=by_seat)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs=2, required=True, type=Path)
    p.add_argument('--mode', choices=['policy', 'one-ply'], required=True)
    p.add_argument('--depth', type=int, choices=[2, 6], default=2)
    p.add_argument('--pairs', type=int, default=256, help='Number of six-ply pairs. At two plies all 720 pairs are used.')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--parallel', type=int, default=64)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if min(a.threads, a.batch, a.parallel, a.pairs) < 1 or a.output.exists():
        p.error('Choose positive batch counts and a fresh output')
    starts = openings(a.depth, a.pairs, a.seed)
    networks = [Network(f, a.threads, a.batch) for f in a.models]
    games, counts = play(starts, networks, a.mode == 'one-ply', a.parallel)
    result = dict(format='uttt-policy-arena-v1', mode=a.mode, openingPlies=a.depth, seed=a.seed,
                  openingSuite='all legal ordered starts' if a.depth == 2 else 'distinct D4 families',
                  tieBreak='lowest legal action index', pairs=len(starts), games=len(starts)*2,
                  networks=[n.sha256 for n in networks], evaluatedPositions=counts, **summarize(games),
                  perPair=games)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=a.output.parent) as temp:
        tmp = Path(temp)/'result'
        tmp.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
        os.link(tmp,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='perPair'}),flush=True)


if __name__ == '__main__':
    main()
