"""Extract public scorecards from completed paired arena reports; write JSON to stdout.

Repeat --match REPORT MODEL_LABEL OPPONENT_LABEL to collect multiple matches.
--identity LABEL SHA256 optionally pins public identities across matches.
Only explicitly supplied labels, notes and stage text enter the output.
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .utttai_stage2 import ZERO, X, O, board_result
else:
    from utttai_stage2 import ZERO, X, O, board_result


FORMATS = {'ultimate-tic-tac-toe-paired-arena-v1': 'native',
           'ultimate-tic-tac-toe-browser-paired-arena-v1': 'browser',
           'official-uttt-arena-v1': 'official'}
OFFICIAL_HASHES = {'81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2',
                   'b4f92e809e1eaa2053b202202d3853e1ed10be6d4c6980c2c46339f6e14f70b7'}
BOOTSTRAP_SEED = 20261126
REPLICATES = 10000
PUBLISHED_SUITE_SHA256 = 'feb92e42ff53472bd8a4a04f4e7d31254c1d87cbdb7ab9f40d110932d001acbe'
# SHA-256 of all 50 upstream 93-digit positions in file order, newline-separated, no final newline.
PUBLISHED_POSITIONS_SHA256 = 'c62514150fccad73b3607d6ccc0dd69a850132d28391d6ddd13da089d037ceef'
PUBLISHED_DEPTHS = {'1': 15, '3': 3, '5': 10, '6': 22}
HARD_ENDGAME = {'maxEmptySquares': 14, 'maxNodes': 16000, 'timeMs': 250}


class ReportError(ValueError):
    """A validation error with a deliberately public-safe message."""


def require(condition, message):
    if not condition:
        raise ReportError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'Expected nonnegative integer count or setting')
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256 identity')
    return value


def public_text(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 .,()%+_-]{0,199}', value)
            and not re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', value),
            'Use a short public label or note without addresses or paths')
    return value


def game_points(game, seat):
    winner = game['winner']
    require(winner in ('X', 'O', 'draw'), 'Invalid game winner')
    value = 0.5 if winner == 'draw' else float(winner == seat)
    require(type(game['score']) in (int, float) and game['score'] == value, 'Seat and game score disagree')
    return value


def interval(clusters):
    # Keep repeated observations of the same ordered opening and both seats together.
    # Divide sampled total points by sampled total games; clusters can differ in size.
    stats = np.array([clusters[key] for key in sorted(clusters)], dtype=np.float64)
    n = len(stats)
    bounds = None
    if n >= 2:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means = np.empty(REPLICATES)
        for start in range(0, REPLICATES, 128):
            stop = min(start + 128, REPLICATES)
            picks = rng.integers(n, size=(stop - start, n))
            sampled = stats[picks].sum(axis=1)
            means[start:stop] = sampled[:, 0] / sampled[:, 1]
        bounds = np.quantile(means, [0.025, 0.975]).tolist()
    return {'method': 'percentile bootstrap by ordered opening, both seats together',
            'distinctOpenings': n, 'replicates': REPLICATES, 'seed': BOOTSTRAP_SEED, 'interval95': bounds}


def opening_state(moves):
    cells, boards = bytearray(b'0' * 81), bytearray(b'0' * 9)
    forced, player = 9, X
    for move in moves:
        require(type(move) is int and 0 <= move < 81, 'Invalid paired opening move')
        board, square = divmod(move, 9)
        require(forced in (9, board) and boards[board] == cells[move] == ZERO, 'Illegal paired opening move')
        cells[move] = player
        boards[board] = board_result(bytes(cells[board * 9:board * 9 + 9]))
        forced = square if boards[square] == ZERO else 9
        player = O if player == X else X
    return (cells + boards + bytes((player, ZERO + forced, ZERO))).decode('ascii')


def timing_summary(raw, cap=None):
    if raw is None:
        return None
    keys = ('min', 'median', 'p95', 'max', 'mean')
    require(isinstance(raw, dict) and all(k in raw for k in ('count', *keys)), 'Incomplete browser timing summary')
    count = integer(raw['count'])
    values = {k: raw[k] for k in keys}
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values.values()),
            'Invalid browser timing statistic')
    if cap is not None:
        require(all(type(values[k]) is int for k in keys[:-1]) and values['max'] <= cap,
                'Invalid browser simulation counts or cap')
    low, median, p95, high, mean = values.values()
    require(low <= median <= p95 <= high and (low <= mean or math.isclose(low, mean))
            and (mean <= high or math.isclose(high, mean)), 'Inconsistent browser timing range')
    if count == 0:
        require(all(v == 0 for v in values.values()), 'Nonzero browser statistics without observations')
        return {'count': 0, **dict.fromkeys(keys)}
    require(count > 1 or low == high, 'Inconsistent single browser observation')
    return {'count': count, **values}


def browser_measured_work(side, timing, cap, rows, opening_plies):
    require(timing.get('label', side) == side and timing.get('pageErrors', []) == [],
            'Browser timing side or page status disagrees')
    simulations = timing_summary(timing.get('simulationCounts'), cap)
    elapsed = timing_summary(timing.get('elapsedMs'))
    counts = [item['count'] for item in (simulations, elapsed) if item is not None]
    if 'moveRequests' in timing:
        counts.append(integer(timing['moveRequests']))
    require(len(set(counts)) <= 1, 'Browser move and timing counts disagree')
    measured = counts[0] if counts else None

    turns = []
    for row in rows:
        for seat in ('X', 'O'):
            game = row['challengerAs' + seat]
            recorded = integer(game[side + 'Turns']) if side + 'Turns' in game else None
            if 'plies' in game:
                game_plies = integer(game['plies'])
                require(game_plies + opening_plies <= 81, 'Browser game plies exceed board size')
                challenger_first = (seat == 'X') == (opening_plies % 2 == 0)
                expected = (game_plies + (challenger_first == (side == 'challenger'))) // 2
                require(recorded is None or recorded == expected, 'Browser game turn count disagrees with seat')
                recorded = expected
            turns.append(recorded)
    if measured is not None:
        known = sum(t for t in turns if t is not None)
        require(known <= measured and (None in turns or known == measured), 'Browser timing count differs from scored turns')

    sources = {k: integer(timing[k]) if k in timing else None
               for k in ('modelMoves', 'verifiedEndgameMoves', 'simulationCapHits')}
    model, exact, hits = sources.values()
    if measured is not None:
        require(all(n <= measured for n in sources.values() if n is not None)
                and (model is None or exact is None or model + exact == measured), 'Invalid browser move-source counts')
    if simulations is not None and simulations['count']:
        require(not exact or simulations['min'] == 0, 'Verified browser endgames must have zero visits')
        require(hits is None or (hits > 0) == (simulations['max'] == cap), 'Browser cap hits disagree with observed visits')
    require(hits is None or model is None or hits <= model, 'Browser cap hits exceed model moves')
    if 'simulationCapHitRate' in timing:
        rate = timing['simulationCapHitRate']
        require(type(rate) in (int, float) and 0 <= rate <= 1 and
                (measured is None or hits is None or math.isclose(rate, hits / measured if measured else 0)),
                'Invalid browser simulation cap-hit rate')
    if measured is None and all(n is None for n in sources.values()):
        return None
    return {'scoredMoves': measured, 'simulations': simulations, 'elapsedMs': elapsed, **sources}


def scorecard(report, model, opponent, identities=None, stage=None, note=None):
    model, opponent = public_text(model), public_text(opponent)
    arena = FORMATS.get(report['format'])
    require(arena is not None, 'Expected a final native, browser or official paired report')
    require(report.get('status', 'complete') == 'complete', 'Arena report is incomplete')
    pairs = integer(report['pairs'], 1)
    seed = integer(report['seed'])
    schedule = report.get('openingSchedule', 'legacy')
    require(schedule in ('legacy', 'independent'), 'Unknown opening schedule')
    opening_suite = report.get('openingSuite', 'generated')
    require(opening_suite == 'generated' or (opening_suite == 'published' and arena == 'official'),
            'Unknown opening suite for this arena')
    published = opening_suite == 'published'
    if published:
        require(pairs == integer(report.get('openingPositions')) == 50 and
                report.get('openingSuiteSha256') == PUBLISHED_SUITE_SHA256 and
                report.get('openingPlyDistribution') == PUBLISHED_DEPTHS, 'Published suite metadata disagrees')
    browser_timed = False
    worker_observations = []
    if arena == 'official':
        require(report['status'] == 'complete', 'Incomplete official match')
        hashes = (digest(report['candidate']['sha256']), digest(report['official']['sha256']))
        require(hashes[1] in OFFICIAL_HASHES, 'Opponent is not a recognized official Stage 2 export')
        require(report['upstreamCommit'] == '622dae66b18198f6f12aef0a0bccd5bd40de2d8e',
                'Unrecognized upstream search revision')
        cap = integer(report['maxRootVisits'], 2)
        opponent_cap = integer(report.get('officialMaxRootVisits', cap), 2)
        require(max(cap, opponent_cap) <= 10000, 'Official match visit cap out of range')
        hard_endgame = report.get('hardEndgame', False)
        require(type(hard_endgame) is bool, 'Invalid Hard endgame setting')
        solver_settings = report.get('hardEndgameSettings')
        require(solver_settings == (HARD_ENDGAME if hard_endgame else None), 'Unexpected Hard endgame limits')
        limits = {'modelSimulations': cap, 'opponentSimulations': opponent_cap, 'modelHardEndgame': hard_endgame}
        if hard_endgame:
            limits['modelEndgameSettings'] = HARD_ENDGAME.copy()
        backend = report.get('candidateBackend', 'native')
        require(backend in ('native', 'chromium-worker'), 'Unknown official-match candidate backend')
        browser_timed = backend == 'chromium-worker'
        require(('browserCandidate' in report) == browser_timed, 'Inconsistent browser worker protocol')
        if browser_timed:
            worker = report['browserCandidate']
            expected = {'difficulty': 'hard', 'simulations': 384, 'timeMs': 220, 'tacticalProof': True}
            require(worker['settings'] == expected and hard_endgame and cap == 384 and integer(report['parallel']) == 1,
                    'Invalid production worker settings or parallelism')
            require(digest(worker['modelSha256']) == hashes[0], 'Browser model differs from candidate')
            limits.update(modelBackend='Chromium production Hard worker', modelTimeMs=220,
                          modelWorkerSha256=digest(worker['workerSha256']), modelDistSha256=digest(worker['distSha256']),
                          opponentBackend='Published NMCTS with fixed root visits')
        rows, summary, plies = report['results'], report, 2
    else:
        hashes = (digest(report['challengerModel']['sha256']), digest(report['championModel']['sha256']))
        rows, summary = report['perPair'], report['challenger']
        require(integer(report['games']) == integer(summary['games']) == pairs * 2, 'Game counts disagree')
        if arena == 'native':
            limits = {'modelSimulations': integer(report['simulations'], 1), 'opponentSimulations': report['simulations']}
            plies = integer(report['openingPlyCount'])
            require(plies <= 8, 'Invalid native opening length')
            for name in ('challengerValueMode', 'championValueMode'):
                if name in report:
                    require(report[name] in ('model', 'zero', 'heuristic'), 'Invalid native value mode')
                    limits[name] = report[name]
            if 'tacticalProof' in report:
                require(type(report['tacticalProof']) is bool, 'Invalid native tactical proof setting')
                limits['tacticalProof'] = report['tacticalProof']
        else:
            plies = integer(report['settings']['openingPlyCount'], 1)
            limits = {}
            for side, label, expected in zip(('challenger', 'champion'), ('model', 'opponent'), hashes):
                settings = report['workerComparison'][f'{side}Settings']
                limits[f'{label}Simulations'] = integer(settings['simulations'], 1)
                limits[f'{label}TimeMs'] = integer(settings['timeMs'], 1)
                timing = report.get('timing', {}).get(side, {})
                require(isinstance(timing, dict), 'Invalid browser timing side')
                if timing:
                    require(digest(timing.get('modelSha256')) == expected, 'Browser model hashes disagree')
                if 'settings' in timing:
                    require(all(timing['settings'].get(k) == v for k, v in settings.items()
                                if k in ('simulations', 'timeMs', 'tacticalProof', 'difficulty')), 'Browser timing settings disagree')
                worker = report['workerComparison'].get(side + 'Sha256')
                if side == 'challenger' and 'workerSha256' in report.get('runtime', {}):
                    runtime_worker = digest(report['runtime']['workerSha256'])
                    require(worker is None or worker == runtime_worker, 'Browser worker hashes disagree')
                    worker = runtime_worker
                limits[label + 'WorkerSha256'] = digest(worker) if worker is not None else None
            require((report['settings']['simulations'], report['settings']['timeMs']) ==
                    (limits['modelSimulations'], limits['modelTimeMs']), 'Browser search settings disagree')
    require(model != opponent or hashes[0] == hashes[1], 'One public label identifies two different models')
    for label, expected in zip((model, opponent), hashes):
        if identities is not None:
            require(identities.get(label, expected) == expected, 'Public label and model SHA-256 disagree')
    offset = 0
    if arena == 'native' and 'arenaShard' in report:
        shard = report['arenaShard']
        offset = integer(shard['pairOffset'])
        require(integer(shard['pairCount']) == pairs and offset + pairs <= integer(shard['plannedPairs']),
                'Invalid native shard boundaries')
    require(len(rows) == len(report['pairScores']) == pairs, 'Pair count disagrees')
    clusters = defaultdict(lambda: [0., 0])
    published_states = []
    depths = Counter()
    wins = draws = losses = exact_moves = retained_moves = 0
    pair_counts = {'wins': 0, 'draws': 0, 'losses': 0}
    for i, row in enumerate(rows):
        require(integer(row['pair']) == offset + i, 'Missing, duplicate, or unordered opening pair')
        opening = row['opening' if arena == 'official' else 'openingMoves']
        require(isinstance(opening, list) and (len(opening) in (1, 3, 5, 6) if published else len(opening) == plies),
                'Invalid paired opening length')
        state = opening_state(opening)
        depth = len(opening)
        if published:
            depths[str(depth)] += 1
            published_states.append(state)
        if arena == 'official':
            games = row['games']
            require(len(games) == 2 and {g['seat'] for g in games} == {'X', 'O'}, 'Missing or duplicate seat')
            require(all(g['moves'][:depth] == opening for g in games), 'Seat games did not use the same opening')
            scores = [game_points(g, g['seat']) for g in games]
            for game in games:
                exact = integer(game.get('exactMoves', None if hard_endgame else 0))
                moves_first = game['seat'] == ('X' if depth % 2 == 0 else 'O')
                model_turns = (len(game['moves']) - depth + moves_first) // 2
                require(exact <= model_turns and (hard_endgame or exact == 0), 'Invalid solver move count for seat or setting')
                exact_moves += exact
                if browser_timed:
                    require('retainedEndgameMoves' in game and game['retainedEndgameMoves'] is None,
                            'Production worker does not report retained-root classification')
                    observed = game['browserTurns']
                    model_moves = game['moves'][depth + (0 if moves_first else 1)::2]
                    require(len(observed) == model_turns, 'Wrong number of scored browser turns')
                    for turn, move in zip(observed, model_moves):
                        visits, elapsed, source = integer(turn['simulations']), turn['elapsedMs'], turn['source']
                        require(integer(turn['move']) == move and visits <= cap and type(elapsed) in (int, float)
                                and math.isfinite(elapsed) and elapsed >= 0 and source in ('model', 'search')
                                and (source != 'search' or visits == 0), 'Invalid browser move observations')
                    require(integer(game['ourVisits']) == sum(t['simulations'] for t in observed)
                            and exact == sum(t['source'] == 'search' for t in observed)
                            and integer(game['theirVisits']) == (len(game['moves']) - depth - model_turns) * opponent_cap,
                            'Browser game visits or verified solver counts disagree')
                    worker_observations.extend(observed)
                else:
                    require('browserTurns' not in game, 'Unexpected worker observations in a fixed-visit report')
                    retained = integer(game.get('retainedEndgameMoves', None if hard_endgame else 0))
                    require(retained <= exact, 'Invalid retained solver move count')
                    retained_moves += retained
        else:
            scores = [game_points(row['challengerAs' + seat], seat) for seat in ('X', 'O')]
        points = sum(scores)
        pair_counts['wins' if points > 1 else 'losses' if points < 1 else 'draws'] += 1
        if 'points' in row:
            require(type(row['points']) in (int, float) and row['points'] == points, 'Incorrect pair point count')
        reported_pair = report['pairScores'][i]
        reported_row = row.get('pairScore', points / 2)
        require(type(reported_pair) in (int, float) and type(reported_row) in (int, float)
                and reported_pair == reported_row == points / 2,
                'Pair scores disagree with seat results')
        cluster = clusters[tuple(opening)]
        cluster[0] += points
        cluster[1] += 2
        wins += scores.count(1)
        draws += scores.count(0.5)
        losses += scores.count(0)
    require([integer(summary[k]) for k in ('wins', 'draws', 'losses')] == [wins, draws, losses],
            'Reported win/draw/loss counts disagree with seat results')
    mean = (wins + 0.5 * draws) / (2 * pairs)
    reported = summary['score' if arena == 'official' else 'scoreRate']
    require(type(reported) in (int, float) and math.isclose(mean, reported, rel_tol=0, abs_tol=1e-12), 'Mean score disagrees')
    if arena == 'native' and 'results' in report:
        details = report['results']
        require(all(integer(details['pairs'][k]) == count for k, count in pair_counts.items())
                and details['pairedScoreRate'] == mean, 'Native pair summary disagrees')
    if published:
        state_digest = hashlib.sha256('\n'.join(published_states).encode('ascii')).hexdigest()
        require(state_digest == PUBLISHED_POSITIONS_SHA256 and dict(depths) == PUBLISHED_DEPTHS,
                'Game prefixes differ from the published positions or their order')
    card = {'arena': arena, 'model': {'label': model, 'sha256': hashes[0]},
            'opponent': {'label': opponent, 'sha256': hashes[1]}, 'seed': seed, 'search': limits,
            'openingPlies': None if published else plies, 'openingSchedule': 'published' if published else schedule,
            'pairOffset': offset, 'pairs': pairs,
            'games': pairs * 2, 'wins': wins, 'draws': draws, 'losses': losses, 'score': mean,
            'uncertainty': ({'method': 'Observed score across the complete fixed published tournament suite',
                             'distinctOpenings': len(clusters), 'interval95': None}
                            if published else interval(clusters))}
    if published:
        card.update(openingSuite='published', openingSuiteSha256=PUBLISHED_SUITE_SHA256, openingPositions=pairs,
                    openingPlyDistribution=dict(depths), pairSeedSchedule=schedule)
    if arena == 'official':
        for field, total in (('exactMoves', exact_moves), ('retainedEndgameMoves', retained_moves)):
            if browser_timed and field == 'retainedEndgameMoves':
                require(field in report and report[field] is None, 'Unexpected retained-root classification')
                card[field] = None
            else:
                count = integer(report.get(field, None if hard_endgame else 0))
                require(count == total, 'Summary solver move count disagrees with seat games')
                card[field] = count
    if arena == 'browser':
        for side, label in (('challenger', 'model'), ('champion', 'opponent')):
            timing = report.get('timing', {}).get(side, {})
            card[label + 'MeasuredWork'] = browser_measured_work(side, timing, limits[label + 'Simulations'], rows, plies)
    if browser_timed:
        require(worker_observations, 'Missing scored browser moves')
        workload = {'scoredMoves': len(worker_observations), 'verifiedEndgameMoves': exact_moves,
                    'modelMoves': len(worker_observations) - exact_moves}
        for key in ('simulations', 'elapsedMs'):
            values = sorted(turn[key] for turn in worker_observations)
            workload[key] = {'min': values[0], 'mean': sum(values) / len(values), 'max': values[-1],
                             'median': values[math.ceil(.5 * len(values)) - 1],
                             'p95': values[math.ceil(.95 * len(values)) - 1]}
        card['modelMeasuredWork'] = workload
    if stage is not None:
        card['stage'] = public_text(stage)
    if note is not None:
        card['note'] = public_text(note)
    if identities is not None:
        identities.update(zip((model, opponent), hashes))
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', nargs=3, metavar=('REPORT', 'MODEL', 'OPPONENT'), action='append', required=True)
    parser.add_argument('--identity', nargs=2, metavar=('LABEL', 'SHA256'), action='append', default=[])
    parser.add_argument('--stage', help='Public stage label for these matches')
    parser.add_argument('--note', help='Explicitly public editorial note for these matches')
    args = parser.parse_args()
    identities, cards = {}, []
    try:
        labels = {label for _, model, opponent in args.match for label in (model, opponent)}
        for label, sha in args.identity:
            require(label in labels, 'Pinned identity has no matching public label')
            require(identities.setdefault(public_text(label), digest(sha)) == sha, 'Conflicting pinned identities')
        for source, model, opponent in args.match:
            report = json.loads(Path(source).read_text(encoding='utf-8'))
            cards.append(scorecard(report, model, opponent, identities, args.stage, args.note))
    except ReportError as error:
        parser.exit(2, f'Scorecard rejected for match {len(cards) + 1}: {error}.\n')
    except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
        # Do not echo file names or arbitrary input fields/exception strings into published output.
        parser.exit(2, f'Scorecard rejected for match {len(cards) + 1} ({type(error).__name__}).\n')
    print(json.dumps({'format': 'uttt-public-arena-scorecards-v1', 'matches': cards}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
