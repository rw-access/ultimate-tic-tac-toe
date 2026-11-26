import copy
import json
import unittest

from training.arena_scorecard import OFFICIAL_HASHES, ReportError, scorecard


A = 'a' * 64
B = sorted(OFFICIAL_HASHES)[0]
MARKER = '/private-host/path/to/checkpoint'
HASHES = {'challengerModel': {'model': MARKER, 'sha256': A},
          'championModel': {'model': MARKER, 'sha256': B}}
WDL = {'wins': 2, 'draws': 2, 'losses': 2}
OPENINGS = ([0, 1], [0, 1], [1, 9])
# Short legal histories for the pinned upstream evaluation positions.
PUBLISHED_OPENINGS = [
    [0], [1], [2], [4], [5],
    [8], [9], [10], [12], [13],
    [15], [16], [36], [37], [40],
    [40, 44, 80, 78, 60], [40, 42, 60, 54, 8], [40, 38, 20, 24, 54], [40, 42, 60, 56, 20], [40, 36, 8, 72, 5],
    [40, 42, 56, 24, 54], [40, 44, 72, 2, 20], [40, 36, 8, 78, 56], [40, 42, 54, 0, 8], [40, 36, 2, 24, 56],
    [40, 38, 23], [40, 44, 76], [40, 38, 21], [10, 16, 70, 66, 32, 48], [38, 19, 11, 18, 5, 51],
    [73, 10, 14, 47, 25, 66], [4, 40, 36, 2, 19, 12], [18, 4, 42, 54, 8, 75], [39, 32, 46, 15, 60, 58], [62, 72, 4, 40, 36, 2],
    [17, 80, 73, 14, 50, 48], [42, 58, 44, 78, 55, 11], [18, 0, 6, 56, 26, 76], [50, 48, 31, 40, 37, 10], [40, 41, 48, 35, 80, 72],
    [37, 9, 3, 33, 59, 52], [79, 65, 19, 11, 18, 0], [24, 58, 43, 64, 15, 59], [16, 65, 24, 59, 48, 27], [60, 58, 43, 64, 17, 80],
    [44, 74, 18, 2, 24, 59], [69, 59, 49, 39, 29, 19], [64, 16, 68, 48, 34, 65], [58, 38, 24, 54, 4, 40], [43, 66, 29, 25, 67, 44],
]
GAMES = ((('X', 1.), ('O', 1.)), (('O', 0.), ('X', 0.)), (('draw', .5), ('draw', .5)))


def row(i):
    games = [{'winner': winner, 'score': score, 'secret': MARKER} for winner, score in GAMES[i]]
    return {'pair': i, 'openingMoves': OPENINGS[i], 'challengerAsX': games[0], 'challengerAsO': games[1],
            'pairScore': sum(g['score'] for g in games) / 2}


def native():
    return {'format': 'ultimate-tic-tac-toe-paired-arena-v1', **HASHES, 'pairs': 3, 'games': 6, 'simulations': 256,
            'openingPlyCount': 2, 'seed': 123, 'perPair': [row(i) for i in range(3)], 'pairScores': [1., 0., .5],
            'challenger': {**WDL, 'games': 6, 'scoreRate': .5}, 'trainingPath': MARKER, 'note': MARKER}


def official():
    rows = []
    for i, opening in enumerate(OPENINGS):
        games = [{'seat': seat, 'winner': winner, 'score': score, 'moves': list(opening) + [42]}
                 for seat, (winner, score) in zip('XO', GAMES[i])]
        rows.append({'pair': i, 'opening': opening, 'games': games[::-1]})
    return {'format': 'official-uttt-arena-v1', 'status': 'complete', **WDL, 'score': .5, 'pairs': 3,
            'candidate': {'file': MARKER, 'sha256': A}, 'official': {'file': MARKER, 'sha256': B},
            'maxRootVisits': 256, 'seed': 123, 'openingSchedule': 'independent',
            'upstreamCommit': '622dae66b18198f6f12aef0a0bccd5bd40de2d8e',
            'pairScores': [1., 0., .5], 'results': rows, 'harnessPath': MARKER}


def browser():
    raw = native()
    raw.update(format='ultimate-tic-tac-toe-browser-paired-arena-v1', settings={'openingPlyCount': 2, 'simulations': 384, 'timeMs': 220},
               workerComparison={'challengerSettings': {'simulations': 384, 'timeMs': 220},
                                 'championSettings': {'simulations': 384, 'timeMs': 650}},
               timing={'challenger': {'modelSha256': A, 'model': MARKER}, 'champion': {'modelSha256': B, 'model': MARKER}},
               runtime={'secret': MARKER})
    return raw


class ScorecardTests(unittest.TestCase):
    def test_native_opening_depth_and_free_choice_after_board_closes(self):
        raw = native()
        for opening in ([], [1, 9, 2, 18, 0, 40]):
            raw['openingPlyCount'] = len(opening)
            for pair in raw['perPair']:
                pair['openingMoves'] = opening
            self.assertEqual(scorecard(raw, 'A', 'B')['openingPlies'], len(opening))
        raw['perPair'][0]['openingMoves'] = [1, 9, 2, 18, 0, 4]
        with self.assertRaises(ReportError):
            scorecard(raw, 'A', 'B')

    def test_public_projection_for_all_harnesses(self):
        for get in (native, browser, official):
            with self.subTest(get=get):
                card = scorecard(get(), 'Candidate', 'Opponent', {'Candidate': A, 'Opponent': B}, 'Screen', 'Public note')
                self.assertTrue(all(card[k] == v for k, v in WDL.items()))
                self.assertEqual(card['score'], .5)
                self.assertEqual(card['uncertainty']['distinctOpenings'], 2)
                # The repeated first opening's winning and losing pairs travel together.
                self.assertEqual(card['uncertainty']['interval95'], [.5, .5])
                self.assertEqual((card['stage'], card['note']), ('Screen', 'Public note'))
                public = json.dumps(card)
                for private in (MARKER, 'openingMoves', 'moves', 'runtime', 'harnessPath'):
                    self.assertNotIn(private, public)
        self.assertEqual(scorecard(browser(), 'A', 'B')['search']['opponentTimeMs'], 650)
        sharded = native()
        sharded.update(arenaShard={'pairCount': 3, 'pairOffset': 4, 'plannedPairs': 8})
        for pair in sharded['perPair']:
            pair['pair'] += 4
        self.assertEqual(scorecard(sharded, 'A', 'B')['pairOffset'], 4)

    def test_browser_h2h_digests_and_measured_work(self):
        raw = browser()
        raw['workerComparison'].update(challengerSha256='c' * 64, championSha256='d' * 64)
        raw['runtime'].update(workerSha256='c' * 64, userAgent=MARKER)
        for row in raw['perPair']:
            for seat in ('X', 'O'):
                row['challengerAs' + seat].update(plies=2, challengerTurns=1, championTurns=1)
        for side in ('challenger', 'champion'):
            raw['timing'][side].update(label=side, moveRequests=6,
                settings=raw['workerComparison'][side + 'Settings'].copy(), pageErrors=[],
                simulationCounts={'count': 6, 'min': 0, 'median': 32, 'p95': 384, 'max': 384,
                                  'mean': 608/6, 'extra': MARKER},
                elapsedMs={'count': 6, 'min': 1, 'median': 100, 'p95': 573, 'max': 573, 'mean': 193},
                simulationCapHits=1, simulationCapHitRate=1/6)
        raw['timing']['challenger'].update(modelMoves=4, verifiedEndgameMoves=2)
        card = scorecard(raw, 'A', 'B')
        self.assertEqual(card['search']['modelWorkerSha256'], 'c' * 64)
        self.assertEqual(card['search']['opponentWorkerSha256'], 'd' * 64)
        for name in ('modelMeasuredWork', 'opponentMeasuredWork'):
            self.assertEqual(card[name]['scoredMoves'], 6)
            self.assertEqual(card[name]['simulations']['median'], 32)
            self.assertEqual(card[name]['elapsedMs']['p95'], 573)
        self.assertEqual(card['modelMeasuredWork']['verifiedEndgameMoves'], 2)
        self.assertIsNone(card['opponentMeasuredWork']['verifiedEndgameMoves'])
        self.assertNotIn(MARKER, json.dumps(card))
        changes = [(('timing','challenger','moveRequests'), 7), (('perPair',1,'challengerAsX','challengerTurns'), 0),
                   (('timing','challenger','simulationCounts','count'), 7), (('timing','challenger','elapsedMs','min'), -1),
                   (('timing','challenger','simulationCounts','max'), 385), (('timing','challenger','modelMoves'), 5),
                   (('timing','challenger','elapsedMs','p95'), 75), (('timing','champion','elapsedMs','mean'), float('nan')),
                   (('timing','champion','elapsedMs','max'), MARKER), (('timing','champion','settings','timeMs'), 220),
                   (('timing','challenger','simulationCapHitRate'), 0.9), (('timing','challenger','pageErrors'), [MARKER]),
                   (('runtime','workerSha256'), 'd' * 64), (('workerComparison','championSha256'), MARKER)]
        for path, value in changes:
            with self.subTest(path=path), self.assertRaises(ReportError):
                changed = copy.deepcopy(raw)
                parent = changed
                for step in path[:-1]:
                    parent = parent[step]
                parent[path[-1]] = value
                scorecard(changed, 'A', 'B')

    def test_browser_absent_empty_and_measured_zero(self):
        raw = browser()
        missing = scorecard(raw, 'A', 'B')
        self.assertIsNone(missing['modelMeasuredWork'])
        self.assertIsNone(missing['search']['modelWorkerSha256'])
        stats = {'count': 0, 'min': 0, 'max': 0, 'median': 0, 'p95': 0, 'mean': 0}
        raw['timing']['challenger']['simulationCounts'] = stats
        empty = scorecard(raw, 'A', 'B')['modelMeasuredWork']
        self.assertEqual(empty['scoredMoves'], 0)
        self.assertIsNone(empty['simulations']['median'])
        self.assertIsNone(empty['elapsedMs'])
        stats['count'] = 6
        measured = scorecard(raw, 'A', 'B')['modelMeasuredWork']
        self.assertEqual(measured['scoredMoves'], 6)
        self.assertEqual(measured['simulations']['median'], 0)
        self.assertIsNone(measured['modelMoves'])

    def test_official_separate_visit_caps_and_hard_endgame(self):
        legacy = scorecard(official(), 'A', 'B')
        self.assertEqual(legacy['search'], {'modelSimulations': 256, 'opponentSimulations': 256,
                                            'modelHardEndgame': False})
        self.assertEqual((legacy['exactMoves'], legacy['retainedEndgameMoves']), (0, 0))
        report = official()
        settings = {'maxEmptySquares': 14, 'maxNodes': 16000, 'timeMs': 250}
        report.update(officialMaxRootVisits=1000, hardEndgame=True, hardEndgameSettings=settings,
                      exactMoves=3, retainedEndgameMoves=1)
        for i, pair in enumerate(report['results']):
            for game in pair['games']:
                game['exactMoves'] = int(game['seat'] == 'X')
                game['retainedEndgameMoves'] = int(game['seat'] == 'X' and i == 0)
        card = scorecard(report, 'A', 'B')
        self.assertEqual(card['search'], {'modelSimulations': 256, 'opponentSimulations': 1000,
                                          'modelHardEndgame': True, 'modelEndgameSettings': settings})
        self.assertEqual((card['exactMoves'], card['retainedEndgameMoves']), (3, 1))
        for changes in ({'officialMaxRootVisits': True}, {'officialMaxRootVisits': 10001},
                        {'hardEndgame': 'true'}, {'hardEndgame': False, 'hardEndgameSettings': None},
                        {'hardEndgameSettings': dict(settings, timeMs=1000)}, {'exactMoves': 2},
                        {'retainedEndgameMoves': 2}):
            with self.subTest(changes=changes), self.assertRaises(ReportError):
                scorecard({**report, **changes}, 'A', 'B')
        # Even a consistent aggregate cannot attribute a solver move to a seat that had no turn.
        report['results'][0]['games'][0]['exactMoves'] = 1
        with self.assertRaises(ReportError):
            scorecard({**report, 'exactMoves': 4}, 'A', 'B')

    def test_timed_worker_official_workload_and_evidence(self):
        raw = official()
        raw.update(maxRootVisits=384, officialMaxRootVisits=1000, parallel=1, hardEndgame=True,
                   hardEndgameSettings={'maxEmptySquares': 14, 'maxNodes': 16000, 'timeMs': 250},
                   candidateBackend='chromium-worker', exactMoves=1, retainedEndgameMoves=None,
                   browserCandidate={'settings': {'difficulty': 'hard', 'simulations': 384, 'timeMs': 220, 'tacticalProof': True},
                                     'modelSha256': A, 'workerSha256': A, 'distSha256': A})
        for i, pair in enumerate(raw['results']):
            for game in pair['games']:
                turns = [dict(move=42, source='search' if i == 0 else 'model', simulations=0 if i == 0 else 37,
                              elapsedMs=212.5)] if game['seat'] == 'X' else []
                game.update(browserTurns=turns, exactMoves=int(bool(turns) and i == 0), retainedEndgameMoves=None,
                            ourVisits=sum(t['simulations'] for t in turns), theirVisits=0 if turns else 1000)
        card = scorecard(raw, 'A', 'B')
        self.assertEqual(card['search']['modelTimeMs'], 220)
        self.assertEqual(card['search']['opponentSimulations'], 1000)
        self.assertEqual(card['modelMeasuredWork']['simulations']['max'], 37)
        self.assertEqual(card['modelMeasuredWork']['elapsedMs']['mean'], 212.5)
        self.assertEqual(card['modelMeasuredWork']['modelMoves'], 2)
        self.assertIsNone(card['retainedEndgameMoves'])
        paths = [(('parallel',), 2), (('browserCandidate', 'settings', 'timeMs'), 650),
                 (('browserCandidate', 'modelSha256'), B), (('candidateBackend',), 'native'),
                 (('results', 0, 'games', 1, 'browserTurns', 0, 'simulations'), 1),
                 (('results', 1, 'games', 1, 'browserTurns', 0, 'move'), 62),
                 (('results', 2, 'games', 1, 'ourVisits'), 99),
                 (('results', 1, 'games', 0, 'theirVisits'), 123)]
        for path, value in paths:
            with self.subTest(path=path), self.assertRaises(ReportError):
                bad = copy.deepcopy(raw)
                parent = bad
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                scorecard(bad, 'A', 'B')

    def test_fixed_published_corpus_and_odd_opening_turn(self):
        raw = official()
        depths = {'1': 15, '3': 3, '5': 10, '6': 22}
        raw.update(pairs=50, wins=0, draws=100, losses=0, score=.5, pairScores=[.5] * 50,
                   openingSuite='published', openingPositions=50, openingPlyDistribution=depths,
                   openingSuiteSha256='feb92e42ff53472bd8a4a04f4e7d31254c1d87cbdb7ab9f40d110932d001acbe',
                   results=[])
        for i, opening in enumerate(PUBLISHED_OPENINGS):
            first_reply = next(m for m in range((opening[-1] % 9) * 9, (opening[-1] % 9) * 9 + 9) if m not in opening)
            games = [dict(seat=seat, winner='draw', score=.5, moves=opening + [first_reply]) for seat in ('X', 'O')]
            raw['results'].append(dict(pair=i, opening=opening, games=games))
        card = scorecard(raw, 'A', 'B')
        self.assertEqual(card['openingSuite'], 'published')
        self.assertEqual(card['openingPlyDistribution'], depths)
        self.assertEqual(card['openingPositions'], 50)
        self.assertEqual(card['pairSeedSchedule'], 'independent')
        self.assertIsNone(card['openingPlies'])
        self.assertEqual(card['uncertainty']['distinctOpenings'], 50)
        self.assertIsNone(card['uncertainty']['interval95'])
        for changes in ({'openingSuiteSha256': A}, {'openingPositions': 49}, {'openingPlyDistribution': {'1': 50}}):
            with self.subTest(changes=changes), self.assertRaises(ReportError):
                scorecard({**raw, **changes}, 'A', 'B')
        changed = copy.deepcopy(raw)
        # Two individually legal one-ply positions in the wrong source order.
        changed['results'][0]['opening'], changed['results'][1]['opening'] = PUBLISHED_OPENINGS[1], PUBLISHED_OPENINGS[0]
        changed['results'][0]['games'], changed['results'][1]['games'] = raw['results'][1]['games'], raw['results'][0]['games']
        with self.assertRaisesRegex(ReportError, 'positions or their order'):
            scorecard(changed, 'A', 'B')
        raw.update(hardEndgame=True, hardEndgameSettings={'maxEmptySquares': 14, 'maxNodes': 16000, 'timeMs': 250},
                   exactMoves=1, retainedEndgameMoves=0)
        for pair in raw['results']:
            for game in pair['games']:
                game.update(exactMoves=0, retainedEndgameMoves=0)
        raw['results'][0]['games'][1]['exactMoves'] = 1
        self.assertEqual(scorecard(raw, 'A', 'B')['exactMoves'], 1)
        raw['results'][0]['games'][1]['exactMoves'] = 0
        raw['results'][0]['games'][0]['exactMoves'] = 1
        with self.assertRaisesRegex(ReportError, 'solver move count for seat'):
            scorecard(raw, 'A', 'B')

    def test_reject_counts_identity_openings_and_progress(self):
        changes = [(native, ('challenger', 'wins'), 1), (native, ('games',), 9), (native, ('pairScores', 0), True),
                   (native, ('perPair', 1, 'pair'), 0), (native, ('perPair', 2, 'challengerAsX', 'score'), 1),
                   (browser, ('timing', 'champion', 'modelSha256'), A), (browser, ('settings', 'timeMs'), 450),
                   (official, ('results', 0, 'games', 0, 'moves'), [1, 9]), (official, ('official', 'sha256'), A),
                   (official, ('status',), 'running'), (official, ('upstreamCommit',), 'bad'),
                   (native, ('challengerModel', 'sha256'), MARKER), (native, ('openingSchedule',), MARKER)]
        for get, path, value in changes:
            with self.subTest(path=path), self.assertRaises(ReportError):
                report = copy.deepcopy(get())
                parent = report
                for step in path[:-1]:
                    parent = parent[step]
                parent[path[-1]] = value
                scorecard(report, 'Candidate', 'Opponent')
        for args in (('Candidate', 'Candidate', None), ('Candidate', 'Opponent', {'Candidate': B}),
                     ('Candidate', MARKER, None), ('Candidate', 'Opponent', None, None, MARKER)):
            with self.subTest(args=args), self.assertRaises(ReportError):
                scorecard(native(), *args)

    def test_seeded_interval_and_single_opening(self):
        raw = native()
        raw['perPair'][1]['openingMoves'] = [4, 36]
        one = scorecard(raw, 'A', 'B')['uncertainty']
        self.assertEqual(one, scorecard(raw, 'A', 'B')['uncertainty'])
        self.assertLess(one['interval95'][0], .5)
        self.assertGreater(one['interval95'][1], .5)
        for pair in raw['perPair']:
            pair['openingMoves'] = [4, 36]
        self.assertIsNone(scorecard(raw, 'A', 'B')['uncertainty']['interval95'])
