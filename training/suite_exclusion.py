"""Identify training rows equivalent under D4 to an uttt.ai tournament opening.

Pass the path to utttpy/selfplay/evaluation_uttt_states.py from upstream commit
622dae66b18198f6f12aef0a0bccd5bd40de2d8e. The source is read as data.
"""

import hashlib
import re
from pathlib import Path

import numpy as np


PUBLISHED_SUITE_SHA256 = 'feb92e42ff53472bd8a4a04f4e7d31254c1d87cbdb7ab9f40d110932d001acbe'
PACKED_ORBITS_SHA256 = '80cd1bc8b547a803c51a1c3cc3aa3c28fab62cd0564104759742ca27deb7ec6f'


def _transforms():
    """Simultaneously transform macro-board and within-board coordinates."""
    grid = np.arange(9).reshape(3, 3)
    for reflection in (grid, np.fliplr(grid)):
        for rotations in range(4):
            yield np.rot90(reflection, rotations).ravel()


def _features(state):
    digits = np.frombuffer(state, np.uint8) - ord('0')
    cells, boards = digits[:81], digits[81:90]
    player, forced, _ = digits[90:]
    opponent = 3 - int(player)
    allowed = (boards == 0) & ((np.arange(9) == forced) | (forced == 9))
    return np.concatenate((
        cells == player, cells == opponent, (cells == 0) & np.repeat(allowed, 9),
        boards == player, boards == opponent, boards == 3, np.arange(10) == forced,
    ))


class Exclusion:
    """A zero-based public opening ID for every rotated or reflected suite state."""

    def __init__(self, suite):
        raw = Path(suite).read_bytes()
        if hashlib.sha256(raw).hexdigest() != PUBLISHED_SUITE_SHA256:
            raise ValueError('Published opening suite hash mismatch')
        records = re.findall(rb'^    "([0-9]{93})",$', raw, re.MULTILINE)
        if len(records) != 50 or len(set(records)) != 50:
            raise ValueError('Expected 50 distinct tournament openings')

        self.by_state = {}
        self.features = {}
        for index, record in enumerate(records):
            digits = np.frombuffer(record, np.uint8)
            for dest in _transforms():
                rotated = digits.copy()
                rotated[(9 * dest[:, None] + dest).ravel()] = digits[:81]
                rotated[81 + dest] = digits[81:90]
                forced = digits[91] - ord('0')
                if forced < 9:
                    rotated[91] = ord('0') + dest[forced]
                state = rotated.tobytes()
                key = np.packbits(_features(state), bitorder='big').tobytes()
                if self.features.setdefault(key, index) != index:
                    raise ValueError('Tournament openings share a symmetry orbit')
                self.by_state[state] = index

        digest = hashlib.sha256(b''.join(sorted(self.features))).hexdigest()
        if digest != PACKED_ORBITS_SHA256:
            raise ValueError('Encoded tournament opening orbit hash mismatch')
        self.max_depth = max(sum(mark != ord('0') for mark in state[:81]) for state in records)

    def identities(self, x):
        """Return an int16 array with the public opening index, or -1 for other rows."""
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 280 or x.dtype.kind not in 'buif':
            raise ValueError('Expected rows of 280 binary features')
        if x.dtype.kind == 'f':
            valid = np.all((x == 0) | (x == 1))
        else:
            valid = x.size == 0 or (x.min() >= 0 and x.max() <= 1)
        if not valid:
            raise ValueError('Expected binary features')
        result = np.full(len(x), -1, np.int16)
        # Deeper states cannot match. Pack just the candidate rows in each input batch.
        shallow = np.flatnonzero(x[:, :162].sum(axis=1, dtype=np.int16) <= self.max_depth)
        packed = np.packbits(x[shallow].astype(np.uint8, copy=False), axis=1, bitorder='big')
        for row, key in zip(shallow, packed):
            result[row] = self.features.get(key.tobytes(), -1)
        return result

    def keep(self, x):
        """Return a boolean mask selecting rows outside all 50 symmetry orbits."""
        return self.identities(x) < 0
