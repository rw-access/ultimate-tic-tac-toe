# Morning model checkpoint, November 26, 2026

**Gen102 scored 57.2% against Gen101 on the primary client and 62.7% on the secondary client.** Against the original Quick engine, its score was 38.0%, compared with 35.0% for Gen101.

Checkpoint report. The frozen browser finalist is 1,476,740 bytes, compared with 1,480,935 for Gen101. The finalist received 1,200 additional training updates starting from Gen101, followed by a legal-cell policy decoder and five phase-dependent policy and value calibrations.

Finalist SHA-256 `9e5348349e6b28f0e35645e73217daf488a57d93eefc6511d12608120cf45b4f`. Incumbent SHA-256 `599b82e961c908cfcd97134d2dd60db25b5b547f45dc6d755d834ef8eb0b8825`.

## Production Hard against Gen101, the shipped predecessor

Each independently seeded opening was played from both seats. A win earns one point and a draw earns half a point. Both sides used the production worker at 384 visits and 220 ms of tree search, plus its bounded endgame solver. The timing columns give finalist and incumbent figures in that order and include every scored turn. Repeated ordered openings stay together in the 95% bootstrap intervals.

| Match | W–D–L | Score | 95% paired interval | Mean visits | Median visits | Median move, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Primary client | 140–306–66 | 57.2% | 54.4%–60.0% | 333.7 / 259.4 | 384 / 272 | 135.8 / 220.7 |
| Secondary client | 94–133–29 | 62.7% | 58.7%–66.5% | 152.7 / 96.0 | 146 / 85 | 221.6 / 222.3 |

The Hard result met the preregistered head-to-head conditions. The primary confirmation used 256 opening pairs and the secondary used 128. The primary score fell below the requested 75% target.

A separate sensitivity analysis grouped rotations and reflections of each opening. Its 95% intervals were 54.25%–60.10% across 95 primary opening families and 58.81%–66.47% across 70 secondary families.

## Original uttt.ai opponent

Quick uses 1,000 visits on the complete published suite of 50 openings. Strong uses 10,000 visits on 16 independently generated openings. The protocol assigns each model both seats on the same openings and pair seeds within each comparison. The visit and timing figures describe the production browser player.

| Match | W–D–L | Score | 95% paired interval | Mean visits | Median visits | Median move, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Finalist vs Quick | 8–60–32 | 38.0% | Complete suite | 329.6 | 384 | 137.2 |
| Gen101 vs Quick | 4–62–34 | 35.0% | Complete suite | 254.3 | 266 | 220.7 |
| Finalist vs Strong | 0–10–22 | 15.6% | 7.8%–23.4% | 335.5 | 384 | 138.1 |
| Gen101 vs Strong | 0–14–18 | 21.9% | 14.1%–29.7% | 249.1 | 253 | 220.8 |

On Quick, the finalist scored +3.0 percentage points relative to Gen101. Across paired openings it scored higher on 16, equally on 22 and lower on 12. As X, finalist and incumbent scored 46.0% and 44.0%. As O, they scored 30.0% and 26.0%.

On Strong, the finalist scored -6.25 percentage points relative to Gen101. Across paired openings it scored higher on 2, equally on 10 and lower on 4. The paired opening-bootstrap 95% interval for the score difference is -17.19 to +3.12 points. The 32-game Strong comparison gives a limited estimate on these generated openings.

## Original engine at equal search budgets

Both native players used a 256-root-visit cap and reused search subtrees. The project player used its own search and immediate tactical replies with its bounded endgame solver disabled. The original opponent used its published NMCTS. This comparison has no browser time limit.

| Match | W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- |
| Finalist vs original NMCTS | 47–117–28 | 54.9% | 50.8%–59.4% |
| Gen101 vs original NMCTS | 38–118–36 | 50.5% | 46.1%–54.9% |

The native finalist took 47 immediate tactical replies. All 4,829 other finalist turns and all 4,874 opponent turns reached 256 root visits.

With the original engine playing the same openings and pair seeds, Gen102 scored +4.43 percentage points relative to Gen101. It did better on 34 paired openings, equally on 38 and worse on 24. The 95% interval for that change was -1.49 to +10.53 points when resampling the 61 rotation/reflection families.

## Weights, decoder and calibration

The equal-work matches compare the learned weights before phase calibration with the incumbent at 256 visits per player. The six-ply match uses a separate seed.

| Match | W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- |
| New weights vs Gen101, 256 visits each | 55–174–27 | 55.5% | 52.0%–58.8% |
| New weights vs Gen101, six-ply starts | 30–138–24 | 51.6% | 48.2%–54.9% |
| Hard replay and ranking vs matched uniform training | 31–124–37 | 48.4% | 44.3%–52.7% |

The two-ply equal-work confirmation favored the trained weights on its sampled openings. These weights are an earlier export than the phase-calibrated browser finalist.

The matched uniform control leaves the incremental benefit of hard-position replay and the ranking loss unresolved. Both models trained for 1,200 updates.

The first two decoder screens reuse the same opening seed on the two clients. They compare the same Gen101 weights with and without the legal-cell rewrite. Additional screens compare learned weights and phase calibration. Their scores concern separate matchups and cannot be added together.

| Match | W–D–L | Score | 95% paired interval | Mean visits | Median visits | Median move, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Gen101 legal-cell decoder vs original, primary | 15–40–9 | 54.7% | 49.2%–60.2% | 336.2 / 266.2 | 384 / 277 | 137.6 / 220.7 |
| Gen101 legal-cell decoder vs original, secondary | 22–37–5 | 63.3% | 56.2%–70.6% | 174.4 / 106.0 | 165 / 94 | 221.5 / 222.1 |
| New weights and decoder vs Gen101 | 18–67–11 | 53.6% | 48.0%–59.4% | 333.6 / 258.5 | 384 / 269 | 142.6 / 220.7 |
| New vs Gen101 weights, both legal-cell decoders | 23–57–16 | 53.6% | 48.0%–59.2% | 334.7 / 334.0 | 384 / 384 | 141.2 / 139.1 |
| Finalist vs Gen101 legal-cell decoder, primary | 15–44–5 | 57.8% | 50.8%–64.5% | 333.1 / 332.5 | 384 / 384 | 136.9 / 142.2 |
| Finalist vs same weights before phase calibration, secondary | 15–36–13 | 51.6% | 43.0%–60.2% | 148.2 / 139.4 | 142 / 135 | 221.7 / 221.8 |
| Two INT8 gates vs original gates, same learned weights | 13–37–14 | 49.2% | 43.8%–54.7% | 127.5 / 119.1 | 120 / 109 | 222.0 / 222.1 |

With Gen101 weights held fixed, the legal-cell decoder increased median visits from 277 to 384 on the primary client and from 94 to 165 on the secondary. The corresponding game scores were 54.7% and 63.3%.

The direct phase-calibration screen spans an even score. The observed result leaves the size of its independent contribution uncertain.

The two-gate INT8 variant increased median search visits in its screen and finished at or below an even score.

A separately preselected Quick run used A-1200 before phase calibration. It scored 38.5% (9 wins, 59 draws, 32 losses) on the 100-game published suite.

### Trainer reproduction

A fresh 1,200-update replay of the public trainer agreed with the original learned weights on the highest-scoring legal move in 4,088 of 4,096 positions. The root-mean-square errors were 0.0092 in legal logits and 0.0016 in value. The replay produced a different file digest. The release uses the original trained checkpoint, SHA-256 `e57b85a97017ba12f63205b5fe604272e3117f16cb289aee22516f91065d7fa5`. Its dense evaluation export has SHA-256 `cee42d0d4d58832a2b3f8c5b12a38dc33c4d9f150cc199ea7bfdf177c4411cba`.

## Offline downloads

| Size class | Previous download | Selected download | Checkpoint decision |
| --- | ---: | ---: | --- |
| 512 KB | 477,280 bytes | 477,280 bytes | Retain predecessor |
| 5 MB | 4,209,067 bytes | 4,113,260 bytes | Promote qualifying candidate |
| 20 MB | 10,014,318 bytes | 10,014,318 bytes | Retain predecessor |
| 25 MB | 19,989,590 bytes | 19,989,590 bytes | Retain predecessor |
| 50 MB | 19,989,590 bytes | 19,989,590 bytes | Retain predecessor |
| 75 MB | 19,989,590 bytes | 19,989,590 bytes | Retain predecessor |

The 25 MB, 50 MB and 75 MB tiers share one preceding checkpoint.

The following finalists were frozen after independent 128-game screens. Each fresh confirmation uses 192 new opening pairs at 256 visits per player. The full-precision result applies to the three classes that share its predecessor.

| Match | W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- |
| Phase-calibrated compact student screen (512 KB) | 28–75–25 | 51.2% | 47.3%–55.1% |
| Phase-calibrated compact student confirmation (512 KB) | 83–232–69 | 51.8% | 49.2%–54.5% |
| Six-bit mixed teacher screen (5 MB) | 22–96–10 | 54.7% | 50.4%–59.3% |
| Six-bit mixed teacher confirmation (5 MB) | 75–257–52 | 53.0% | 50.3%–55.7% |
| Half-precision averaged teacher screen (20 MB) | 18–99–11 | 52.7% | 48.8%–56.7% |
| Half-precision averaged teacher confirmation (20 MB) | 71–251–62 | 51.2% | 48.7%–53.6% |
| Full-precision averaged teacher screen (25 MB, 50 MB, 75 MB) | 18–97–13 | 52.0% | 48.8%–54.8% |
| Full-precision averaged teacher confirmation (25 MB, 50 MB, 75 MB) | 70–251–63 | 50.9% | 48.5%–53.2% |

The 512 KB qualifier missed the prespecified lower-bound requirement on its fresh openings.

The 5 MB qualifier cleared the prespecified lower-bound requirement on its fresh openings.

The 20 MB qualifier missed the prespecified lower-bound requirement on its fresh openings.

The 25 MB, 50 MB, 75 MB qualifier missed the prespecified lower-bound requirement on its fresh openings.

The following offline diagnostics use 256 visits per side. A promotion requires a fresh confirmation against the tier’s own released checkpoint, a lower 95% score bound above 50% and a download within the former byte size.

| Match | W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- |
| Ranked compact learner vs 512 KB | 19–53–24 | 47.4% | 40.3%–54.7% |
| Compact INT8 scale learner, 5,400 updates vs 512 KB | 28–72–28 | 50.0% | 44.4%–55.7% |
| Compact learner, 6,000 updates vs 512 KB | 28–74–26 | 50.8% | 45.7%–55.8% |
| Extended action-value teacher, four-bit vs 5 MB | 17–65–14 | 51.6% | 46.4%–56.8% |
| Extended action-value teacher, half precision vs 20 MB | 13–71–12 | 50.5% | 45.4%–55.7% |
| Extended action-value teacher, full precision vs 25 MB | 12–72–12 | 50.0% | 45.2%–54.8% |
| Calibrated four-bit teacher vs 5 MB | 14–94–20 | 47.7% | 44.8%–50.4% |
| Calibrated full-precision teacher vs 25 MB | 17–91–20 | 48.8% | 47.2%–50.4% |

## Optional historical reference

Gen69 is an earlier model from the project archive. Gen101 is the shipped predecessor used for both promotion matches above. These historical games use the current production Hard worker on both sides. Prior-release rows use separate opening samples.

| Match | W–D–L | Score | 95% paired interval | Mean visits | Median visits | Median move, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Finalist vs Gen69 | 89–87–16 | 69.0% | 64.0%–74.0% | 338.8 / 340.5 | 384 / 384 | 143.4 / 140.9 |
| Gen101 vs Gen69, previous primary confirmation | 85–87–20 | 66.9% | 62.2%–71.6% | 289.1 / 346.8 | 310 / 384 | 220.4 / 113.9 |
| Gen101 vs Gen69, previous secondary confirmation | 43–67–18 | 59.8% | 53.1%–66.2% | 113.9 / 211.9 | 99 / 211 | 221.7 / 221.1 |

Historical opponent SHA-256 `b6f888ca2a7c987e3366ba2272e01e90f01e7799d74588ed1921c9a2e473485d`.

The published Quick score covers a fixed set of tournament positions. The independent Hard confirmations measure performance over separately sampled starts. The incumbent retains its earlier training history.

[Validated scorecards](artifacts/model-strength-2026-11-26-morning.json)

## Policy play without tree search

Over all 720 legal two-move starts played in both seats, Gen102 scored 47.1% against the original network on greedy policy alone. The full teacher scored 50.5%. At one ply, Gen102 scored 42.7% and the teacher scored 50.8%. In a warmed single-thread browser, a complete greedy move took a median 0.670 ms for Gen102 and 4.260 ms for the original engine. [Teacher and student study, including solved positions and one-ply timings](policy-networks-2026-11-26.md)

## Afternoon strategy

Gen102 becomes the browser comparator for the afternoon. The 5 MB comparator is the six-bit winner. Each other size class uses the exact file in the morning release. The student’s weakest network-only result was choosing among the values of legal replies. One training experiment therefore labels all 229,795 legal successors of 32,334 training roots, and holds the shared policy weights fixed. A separate student will learn from dedicated Gen102 games against the original engine. Opening families reserved for testing stay out of that fit. The Full experiments test whether learning the ordering of replies or combining corresponding layers improves the released teacher.

Freeze one browser finalist before the afternoon confirmation seeds. Use independently seeded, seat-swapped Hard matches on both clients (seeds `26316201` and `26316202`) and repeat the matched published Quick suite. For offline candidates, reserve seed `26316204` for confirmation against each candidate’s own morning release. Use the morning file sizes as caps, the production worker with a 384-visit maximum and 220 ms of tree search, and 256 visits per player in offline matches.
