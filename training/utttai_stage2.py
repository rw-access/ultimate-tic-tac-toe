"""Convert extracted uttt.ai stage2 NMCTS text to sharded 280-feature training data.

The upstream state layout and target convention are defined in utttai's
utttpy/game/constants.py and utttpy/selfplay/datatools.py. Each root value
already uses the side-to-move perspective. The action index is board * 9 + cell.
"""

import argparse
import glob
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


RECORD = re.compile(rb"evaluatedState\{([0-9]{93})\s+([0-9]+)\s+([^}\s]+)\}\s*evaluatedActions\{([^{}]+)\}")
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
ZERO, X, O, DRAW = b"0123"
CELL_MASKS = {mark: bytes(int(i == mark) for i in range(256)) for mark in (X, O, DRAW)}


@lru_cache(maxsize=65536)
def board_result(cells):
    winner = ZERO
    for a, b, c in LINES:
        mark = cells[a]
        if mark in (X, O) and mark == cells[b] == cells[c]:
            if winner != ZERO and winner != mark:
                raise ValueError("board has two winners")
            winner = mark
    return DRAW if winner == ZERO and ZERO not in cells else winner


def bounded_value(token):
    value = float(token)
    if not -1 <= value <= 1:
        raise ValueError("value must be finite and between -1 and 1")
    return value


def parse_record(line):
    match = RECORD.fullmatch(line.strip())
    if match is None:
        raise ValueError("expected a stage2 evaluatedState and evaluatedActions record")
    state, root_visits, root_value, actions = match.groups()
    cells, results = state[:81], state[81:90]
    player, forced_digit, result = state[90:]
    forced = forced_digit - ZERO
    if player not in (X, O) or result != ZERO or max(cells) > O:
        raise ValueError("invalid marks, player, or terminal state")
    if cells.count(X) - cells.count(O) != int(player == O):
        raise ValueError("mark counts disagree with player to move")
    if any(board_result(cells[9*i:9*i+9]) != r for i, r in enumerate(results)):
        raise ValueError("sub-board results disagree with the cells")
    if board_result(results) != ZERO or (forced < 9 and results[forced] != ZERO):
        raise ValueError("closed game or closed forced board")
    opponent = O if player == X else X
    legal = bytearray(81)
    boards = range(9) if forced == 9 else (forced,)
    for board in boards:
        if results[board] == ZERO:
            start = board * 9
            for i in range(start, start + 9):
                legal[i] = int(cells[i] == ZERO)
    visits = np.zeros(81, dtype=np.float32)
    seen = set()
    total = 0
    for action in actions.split(b","):
        symbol, index_field, count_field, action_value = action.split()
        index = int(index_field)
        count = int(count_field)
        if symbol != bytes((player,)) or not (0 <= index < 81 and legal[index]):
            raise ValueError("invalid action symbol or illegal move")
        if index in seen or count < 0:
            raise ValueError("duplicate action or negative visit count")
        bounded_value(action_value)
        visits[index] = count
        seen.add(index)
        total += count
    if len(seen) != sum(legal) or not (0 < total <= int(root_visits)) or not np.isfinite(visits).all():
        raise ValueError("incomplete legal actions or invalid visit counts")
    features = (cells.translate(CELL_MASKS[player]) + cells.translate(CELL_MASKS[opponent])
                + legal + results.translate(CELL_MASKS[player]) + results.translate(CELL_MASKS[opponent])
                + results.translate(CELL_MASKS[DRAW]) + bytes(int(i == forced) for i in range(10)))
    return np.frombuffer(features, np.uint8), visits / total, bounded_value(root_value)


def source_paths(inputs):
    paths = set()
    for pattern in inputs:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches:
            raise ValueError(f"no input matches {pattern}")
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.update(p.resolve() for p in path.rglob("*.txt") if p.is_file())
            elif path.is_file():
                paths.add(path.resolve())
    if not paths:
        raise ValueError("no text files found")
    return sorted(paths)


def sample_counts(counts, maximum):
    total = sum(counts)
    if maximum is None or maximum >= total:
        return counts.copy()
    quotas = [divmod(count * maximum, total) for count in counts]
    result = [q for q, _ in quotas]
    for i in sorted(range(len(quotas)), key=lambda i: -quotas[i][1])[:maximum-sum(result)]:
        result[i] += 1
    return result


def convert(inputs, output, maximum=None, seed=1, shard_size=65536):
    if shard_size < 1 or (maximum is not None and maximum < 1):
        raise ValueError("shard size and optional sample limit must be positive")
    paths = source_paths(inputs)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = []
    if maximum is not None:
        for path in paths:
            with path.open("rb") as f:
                counts.append(sum(1 for line in f if line.strip()))
    quotas = sample_counts(counts, maximum) if maximum is not None else [None] * len(paths)
    rng = random.Random(seed)
    x = np.empty((shard_size, 280), dtype=np.uint8)
    p = np.empty((shard_size, 81), dtype=np.float16)
    v = np.empty(shard_size, dtype=np.float32)
    sources, shards = [], []
    used = 0
    with TemporaryDirectory(dir=output.parent, prefix=".stage2-") as temporary:
        staging = Path(temporary) / "converted"
        staging.mkdir()

        def flush():
            name = f"shard-{len(shards):05d}.npz"
            np.savez_compressed(staging / name, x=x[:used], p=p[:used], v=v[:used])
            shards.append({"file": name, "positions": used})

        for i, (path, quota) in enumerate(zip(paths, quotas)):
            selected = None
            if quota is not None and quota < counts[i]:
                selected = set(rng.sample(range(counts[i]), quota))
            available = sampled = 0
            with path.open("rb") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    index = available
                    available += 1
                    if selected is not None and index not in selected:
                        continue
                    try:
                        x[used], p[used], v[used] = parse_record(line)
                    except (ValueError, OverflowError) as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
                    sampled += 1
                    used += 1
                    if used == shard_size:
                        flush()
                        used = 0
            if quota is not None and (available != counts[i] or sampled != quota):
                raise ValueError(f"{path} changed during conversion")
            sources.append({"file": str(path), "available": available, "sampled": sampled})
        if used:
            flush()
        if not shards:
            raise ValueError("no positions to convert")
        manifest = {"format": "utttai-stage2-v1", "seed": seed, "max_positions": maximum,
                    "positions": sum(s["positions"] for s in shards), "sources": sources, "shards": shards}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="text file, directory, or recursive glob")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-positions", type=int, help="sample a proportional share from each source file")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=65536)
    args = parser.parse_args()
    manifest = convert(args.sources, args.output, args.max_positions, args.seed, args.shard_size)
    print(f"Converted {manifest['positions']:,} positions into {len(manifest['shards'])} shards at {args.output}")


if __name__ == "__main__":
    main()
