# Afternoon model checkpoint, November 26, 2026

Gen102 keeps the browser slot because the primary 95% interval reached an even score. The Tiny download scored 56.5% against its morning predecessor in 384 games and qualified at 477,280 bytes. The 5 MB download scored 53.8% against its morning predecessor in 384 games and qualified at 4,113,260 bytes.

With 220 ms for each side, Gen102 scored **71.0%** against the original uttt.ai engine on the primary browser and **77.0%** on the second. Against the full 1,000-visit Quick setting, its score was **38.0%** across the same 50 openings in both seats. It scored **15.6%** against 10,000-visit Strong over 32 games.

With a single greedy policy evaluation each, Gen102 scored **46.7%** and the morning full teacher scored **51.6%** against the original network. Each played both seats on 4,096 fresh opening families, or 8,192 games.

When every player evaluated all legal moves to depth one, Gen102 scored **42.3%** and the full teacher scored **51.2%** over the same 8,192 games per matchup.

The highest fresh primary score against an immediate predecessor was 56.5% for the 512 KB finalist. The requested target was 75%.

The browser finalist was the student trained on 8,192-visit pilot opponent positions (`r2-j4600-cell-sharedphase`, SHA-256 `4b88127c388b533e6283ecb2d7fc18f360bc180e4d787a7addf9767a29e65cd2`). Its immediate predecessor was the shipped Gen102, SHA-256 `9e5348349e6b28f0e35645e73217daf488a57d93eefc6511d12608120cf45b4f`.

The models played both seats on each opening. A win earns one point and a draw earns half. The comparison for every size class uses its exact morning release. [Morning checkpoint and model hashes](model-strength-2026-11-26-morning.md)

## Browser Hard and the original opponent

| Match | W–D–L | Games | Score | 95% paired interval |
| --- | --- | --- | --- | --- |
| Finalist vs Gen102, primary client | 80–368–64 | 512 | 51.6% | 48.9%–54.1% |
| Finalist vs Gen102, second client | 50–164–42 | 256 | 51.6% | 48.0%–55.1% |
| Finalist vs Gen102 at 256 visits each | 38–181–37 | 256 | 50.2% | 46.9%–53.5% |
| Finalist vs original Quick, 1,000 visits | 6–68–26 | 100 | 40.0% | Complete suite |
| Gen102 vs original Quick, 1,000 visits | 8–60–32 | 100 | 38.0% | Complete suite |
| Finalist vs original Strong, 10,000 visits | 0–12–20 | 32 | 18.8% | 12.5%–25.0% |
| Gen102 vs original Strong, 10,000 visits | 0–10–22 | 32 | 15.6% | 7.8%–23.4% |
| Finalist vs original NMCTS, 256 visits each | 46–121–25 | 192 | 55.5% | 51.5%–59.4% |
| Gen102 vs original NMCTS, 256 visits each | 47–117–28 | 192 | 54.9% | 50.8%–59.4% |

Hard allows 384 visits and 220 ms of tree search plus the existing bounded solver. Quick uses the 50 published starts in both seats. Strong uses 16 independently generated starts in both seats. The equal-work games have no browser deadline.

| Client | Finalist median visits | Gen102 median visits | Finalist median move, ms | Gen102 median move, ms |
| --- | --- | --- | --- | --- |
| Primary | 384 | 384 | 135.5 | 138.1 |
| Second | 161 | 162 | 221.4 | 221.4 |

### Earlier shipped generations

| Afternoon finalist against | W–D–L | Games | Hard score | 95% paired interval |
| --- | --- | --- | --- | --- |
| Gen69 | 95–82–15 | 192 | 70.8% | 66.3%–75.3% |
| Gen101 | 37–130–25 | 192 | 53.1% | 49.0%–57.0% |

Gen69 was the browser model deployed before the overnight training began. Gen101 shipped overnight, followed by Gen102 in the morning. The historical games used the primary browser client, both seats and the same search settings. Gen102 remained the direct afternoon promotion opponent.

| Against original | Matched openings | Finalist minus Gen102, points | 95% interval, points | Better / equal / worse openings |
| --- | --- | --- | --- | --- |
| Quick | 50 | +2.0 | Complete published suite | 15 / 25 / 10 |
| Strong | 16 | +3.1 | -4.7 to +10.9 | 5 / 8 / 3 |
| Equal work | 96 | +0.5 | -4.2 to +5.5 | 22 / 48 / 26 |

Each difference compares the two networks against the same opponent, seats and opening seeds. The interval resamples complete rotation and reflection families. Quick covers the complete published tournament.

### How much original-engine search matches browser Hard?

| Original visits | Gen102 W–D–L | Gen102 score | Finalist W–D–L | Finalist score |
| --- | --- | --- | --- | --- |
| 256 | 28–60–12 | 58.0% | 25–66–9 | 58.0% |
| 384 | 24–59–17 | 53.5% | 22–64–14 | 54.0% |
| 512 | 13–63–24 | 44.5% | 14–63–23 | 45.5% |
| 1000 | 8–60–32 | 38.0% | 6–68–26 | 40.0% |

These are complete 100-game tournaments on the same 50 published openings, with the project’s real Hard worker, deadline, solver and visit cap fixed. Only the original engine’s cap varies. Its website Quick setting corresponds to 1,000 visits. The 256-, 384- and 512-visit settings are diagnostic.

### Both engines in Chromium with a 220 ms search deadline

| Browser client | Project model vs original | W–D–L | Score | Median root visits, project / original | Median move ms, project / original |
| --- | --- | --- | --- | --- | --- |
| Primary | Gen102 | 49–44–7 | 71.0% | 384 / 84 | 111.9 / 222.0 |
| Primary | Finalist | 40–56–4 | 68.0% | 384 / 81 | 122.5 / 222.3 |
| Second | Gen102 | 60–34–6 | 77.0% | 169 / 29 | 221.3 / 226.4 |
| Second | Finalist | 58–38–4 | 77.0% | 165 / 29 | 221.3 / 226.0 |

On each client both players used the same Chromium instance and single-thread WebAssembly runtime. Both clients used identical openings and pair seeds. Hard used its shipped 384-visit limit, 220 ms tree-search deadline and bounded endgame solver. The original Stage 2 engine used its website weights, tree reuse, 1,000-visit Quick cap and an added 220 ms search deadline. It completed any simulation already in flight when the clock expired. The fixed-visit Quick tournament above has no such time cutoff.

Both players took each seat on all 50 published starts. Three fixed-visit trees checked the original browser implementation against the pinned native reference before scoring. A separate replay verified every scored move and final outcome. [Same-clock browser arena](../scripts/timed-official-arena.ts)

### Which late-game moves cost a proved win or draw?

| Browser | Original opponent | Decisions proved / eligible | Win to draw | Win to loss | Draw to loss | Drawn games with a missed win |
| --- | --- | --- | --- | --- | --- | --- |
| Gen102 | Quick | 259 / 259 | 3 | 0 | 4 | 2 / 60 |
| Gen102 | Strong | 89 / 90 | 0 | 0 | 1 | 0 / 10 |
| Finalist | Quick | 254 / 255 | 1 | 0 | 4 | 1 / 68 |
| Finalist | Strong | 84 / 86 | 0 | 0 | 0 | 0 / 12 |

The solver replayed actual browser decisions with two or more legal moves and 15–24 playable squares left, outside the production solver horizon. For every proved position, it classified every legal move. The last column counts each game once if the browser passed up a guaranteed win during this part of the game. The table concerns decisions along the recorded game, so multiple mistakes can occur in one game.

## Size-class decisions

| Size class | Bytes | Decision | Challenger | Fresh W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- | --- | --- | --- |
| 512 KB | 477,280 | Promote | Refitted Tiny INT8 scales | 100–234–50 | 56.5% | 53.7%–59.3% |
| Web | 1,476,740 | Retain | Student trained on 8,192-visit pilot opponent positions | 80–368–64 | 51.6% | 48.9%–54.1% |
| 5 MB | 4,113,260 | Promote | Full teacher trained on its own original-engine games | 72–269–43 | 53.8% | 51.2%–56.4% |
| 20 MB | 10,014,318 | Retain | Own-game full teacher with the previous value layers | 57–280–47 | 51.3% | 48.8%–53.8% |
| 25 MB | 19,989,590 | Retain | Full teacher trained on its own original-engine games | 59–273–52 | 50.9% | 48.1%–53.7% |
| 50 MB | 19,989,590 | Retain | Full teacher trained on its own original-engine games | 59–273–52 | 50.9% | 48.1%–53.7% |
| 75 MB | 19,989,590 | Retain | Full teacher trained on its own original-engine games | 59–273–52 | 50.9% | 48.1%–53.7% |

The 25 MB, 50 MB and 75 MB tiers share one full-weight file. Their repeated result counts as one match. A dash marks a tier with no completed fresh confirmation. Offline confirmation uses 256 visits per side and requires the lower endpoint of the paired 95% interval to exceed 50%.

Resampling complete rotation and reflection families gives a second check on the fresh scores. The public evidence includes these intervals and the number of distinct families for each match.

### A separate six-move opening sample

| Finalist vs morning predecessor | W–D–L | Games | Score | 95% paired interval |
| --- | --- | --- | --- | --- |
| Tiny | 32–112–48 | 192 | 45.8% | 41.7%–50.0% |
| Browser weights | 20–138–34 | 192 | 46.4% | 42.7%–50.3% |

Each matchup used 96 separately seeded six-move opening pairs, both seats and 256 visits per side. This checks whether strength carries over to deeper starts at fixed search work.

### Investigating the Tiny result on longer openings

The scheduled 384-game Tiny confirmation began after two moves and favored the new weights. A separate 192-game sample starting after six moves scored 45.8% for the challenger (32 wins, 112 draws, 48 losses, 95% interval 41.7% to 50.0%). The two runs also used different inference providers. Before running follow-ups, the release decision was made conditional on at least an even score across 768 new six-move games and a paired check against the original engine for a regression on longer starts.

| Finalist vs morning predecessor | W–D–L | Games | Score | 95% paired interval |
| --- | --- | --- | --- | --- |
| Tiny, repeat the primary two-move starts on GPU | 108–226–50 | 384 | 57.6% | 54.5%–60.7% |
| Tiny, repeat the first six-move starts on CPU | 32–114–46 | 192 | 46.4% | 42.2%–50.3% |
| Tiny, new six-move starts on CPU | 131–527–110 | 768 | 51.4% | 49.5%–53.1% |
| 5 MB, new six-move starts on GPU | 61–270–53 | 384 | 51.0% | 48.4%–53.6% |

| Same opening seeds and inference setting | Games per depth | Two-move score | Six-move score | Six minus two, points | 95% paired interval |
| --- | --- | --- | --- | --- | --- |
| Primary two-move sample, GPU | 384 | 57.6% | 51.7% | -5.9 | -11.7 to -0.2 |
| Initial six-move sample, CPU | 192 | 54.2% | 46.4% | -7.8 | -14.5 to -0.8 |

Each six-move start extends its paired two-move start by exactly four legal moves. Both models played both seats at each depth. The intervals keep all rotations, reflections and repetitions of the same two-move start together.

| Opening depth | Matched games | GPU minus CPU score, points | 95% family-paired interval | Identical game outcomes |
| --- | --- | --- | --- | --- |
| Two moves | 384 | +1.0 | -0.4 to +3.0 | 370 / 384 |
| Six moves | 192 | -0.5 | -2.3 to +1.0 | 185 / 192 |

Both exact Tiny files opened directly with the CUDA provider. In 13 single-position and batched fixture evaluations, CPU and GPU selected the same legal move 13 times for the incumbent and 11 times for the finalist. The finalist's disagreements were the same fixture evaluated twice. Its competing moves differed by at most 0.00000119 on CPU and tied on GPU. The largest absolute logit differences across providers were 0.00570 and 0.00813, respectively.

Pooling the two independent six-move samples on CPU gives 163–641–156 over 960 games, a score of 50.4% (95% interval 48.7% to 52.0%, resampling 480 opening families). Each opening pair is counted once across the two samples. Tiny's largest measured advantage was on two-move starts.

| Six-move games vs original | Morning W–D–L | Morning score | Finalist W–D–L | Finalist score | Change, points (95%) |
| --- | --- | --- | --- | --- | --- |
| Tiny | 28–115–49 | 44.5% | 29–124–39 | 47.4% | +2.9 (-2.1 to +7.6) |
| 5 MB | 50–118–24 | 56.8% | 54–121–17 | 59.6% | +2.9 (-2.6 to +8.2) |

Both players received 256 visits. All four files played both seats on the same 96 six-move starts against the original engine. The intervals resample its 95 rotation/reflection families. An independent rules implementation replayed all 768 games.

The Tiny challenger qualified after scoring 51.4% in the new six-move head-to-head. Its original two-move confirmation score was 56.5%.

### Offline downloads against the original engine

| Size class | Morning W–D–L | Morning score | Finalist W–D–L | Finalist score | 95% finalist interval |
| --- | --- | --- | --- | --- | --- |
| 512 KB | 23–106–63 | 39.6% | 35–117–40 | 48.7% | 44.3%–53.1% |
| 5 MB | 42–133–17 | 56.5% | 48–126–18 | 57.8% | 53.7%–62.0% |
| 20 MB | 44–132–16 | 57.3% | 41–137–14 | 57.0% | 53.2%–60.9% |
| Full (25, 50 and 75 MB) | 44–132–16 | 57.3% | 46–129–17 | 57.6% | 53.4%–61.9% |

| Size class | Finalist minus morning against original, points | 95% paired interval, points | Better / equal / worse opening pairs |
| --- | --- | --- | --- |
| 512 KB | +9.1 | +3.8 to +14.2 | 48 / 30 / 18 |
| 5 MB | +1.3 | -4.1 to +6.4 | 26 / 47 / 23 |
| 20 MB | -0.3 | -4.8 to +4.4 | 16 / 59 / 21 |
| Full (25, 50 and 75 MB) | +0.3 | -4.1 to +4.7 | 19 / 57 / 20 |

Both sides received 256 visits. Every file played the same 96 independent two-move starts from both seats against the original uttt.ai Stage 2 NMCTS engine. A matching hash allowed reuse of the existing result for an identical morning download. [Earlier scorecards](artifacts/model-strength-2026-11-26.json)

### Against the models from before the overnight training

| Afternoon download vs its pre-overnight predecessor | W–D–L | Games | Score | 95% interval |
| --- | --- | --- | --- | --- |
| 512 KB | 237–16–3 | 256 | 95.7% | 93.6%–97.5% |
| 5 MB | 216–35–5 | 256 | 91.2% | 88.6%–93.7% |
| 20 MB | 165–78–13 | 256 | 79.7% | 75.8%–83.5% |
| 25 MB | 180–63–13 | 256 | 82.6% | 79.1%–86.1% |
| 50 MB | 134–108–14 | 256 | 73.4% | 69.6%–77.5% |
| 75 MB | 133–98–25 | 256 | 71.1% | 66.6%–75.4% |

Each size class faced the archived model for that class at 256 visits per side. An identical afternoon and overnight file uses its existing match. The 25 MB, 50 MB and 75 MB entries share one current model and have three different archived opponents. [Archived opponents and original scorecards](artifacts/model-strength-2026-11-26.json)

The 5 MB finalist also scored 52.0% (64–271–49, 384 games, 95% interval 49.1%–54.8%) against the four-bit model released overnight. The primary promotion opponent was the six-bit model that shipped in the morning.

## What training changed

One experiment used 32,334 training roots and all 229,795 legal successor positions. Exact terminal outcomes, the original network in eight orientations and search-supported action values supplied the training targets. The student’s policy stayed fixed while its value head learned to order the replies. A second experiment used the complete distribution of original root visits when training those values. The Full teacher also trained on the all-reply targets.

The released full teacher played 160 dedicated training games against the original engine, using 512 and 2,048 visits per move, respectively. An independent 8,192-visit search relabeled its decisions. The trainer used 1,454 distinct positions after exclusions and sampled 91 supported mistakes more often. It averaged the weights during 2,400 updates interleaved with exact and general positions.

On 368 positions from 11 held opening families, the full model’s policy cross-entropy changed from 1.0926 to 1.0867 and its value mean squared error increased from 0.1105 to 0.1191.

Restoring the predecessor’s value layers on the trained body brought value error to 0.1174. The move probabilities on the held positions stayed the same.

The strongest early Tiny screen came from refitting 35,686 scale factors on fixed signed eight-bit weights. Each training batch used 50 boards from Tiny’s own original-engine games and 110 from the general data. On 458 held positions from its own games, the 1,200-update checkpoint lowered policy cross-entropy from 1.317 to 1.280 and value mean squared error from 0.115 to 0.090. It retained the 477,280-byte graph.

Starting from the released Tiny file, running the public trainer again produced a byte-for-byte identical candidate.

A replay with the public trainer reproduced the same highest-rated legal move on 2,060 of 2,060 parity and replay positions. All changes were confined to 20 scale tensors. [Tiny scale trainer](../training/refit_tiny_scales.py)

A separately trained control substituted general self-play boards matched by game phase and legal-move count, with the same 8,192-visit reference, initialization, seed, 1,200 optimizer steps and batch mix. Tiny trained on its own games scored 49.7% against that control (41–109–42 over 192 games, 95% interval 46.3%–53.2%).

The general-data control then scored 53.8% against the morning Tiny on the same 384 confirmation games used for the finalist (105–203–76, 95% interval 50.3%–57.5%). The control was reserved for this comparison and did not enter finalist selection.

On those identical confirmation openings, the Tiny trained with its own opponent-game boards scored +2.7 points relative to the general-data control against the same incumbent (95% family-paired interval -3.8 to +9.1).

The compact 512 KB experiment updated 144,238 of 285,488 existing signed eight-bit weights. Each forward training pass used integer rounding. Each weight moved at most 6 integer levels, and the serialized graph kept the predecessor’s byte count.

In a separate scale-training experiment, the morning 5 MB release played 96 dedicated training games at 256 visits against the original engine at 1,024 visits. Deep searches at 4,096 visits relabeled its moves. A 1,800-update experiment trained four policy-output scale tensors at fixed six-bit integer weights, mixing 990 unique retained positions from its games with positions from the full teacher and general play.

On 318 positions from separate opening families, policy cross-entropy went from 1.1586 to 1.1464. The exported value computation stayed identical.

Another experiment started from the morning 5 MB download. It optimized the ten policy and value output layers with exact six-bit rounding on each training pass. It changed 393,453 of 988,416 integer codes in those layers by up to two levels and kept all 4,113,260 serialized bytes of budget.

A separate student learned from 148 moves actually played against the original engine for which deep search supported a better alternative. The trainer updated only the policy head for 2,200 steps and verified identical values on 396 parity boards.

| Held opponent games | Eligible positions | Opening families | Policy loss, Gen102 / trained | Value error, Gen102 / trained | Top search move, Gen102 / trained |
| --- | --- | --- | --- | --- | --- |
| 10,000-visit original opponent | 96 | 4 | 1.2423 / 1.2265 | 0.0800 / 0.0570 | 63.5% / 67.7% |
| 1,536-visit original opponent | 652 | 14 | 1.0754 / 1.0687 | 0.0834 / 0.0799 | 65.6% / 67.3% |

A separate compact student trained for 2,600 updates on decisions from newly generated games. Its exported weights were frozen before the held 8,192-visit targets could be read. The evaluation also excluded positions and immediate successors found in the registered parent training history. [Preparation and evaluation](../training/opponent_student.py)

Exact replay on five dedicated training series examined choices with 15–28 playable squares left. A position-value solver proved the original and played outcomes on 1,692 positions, and verified a better legal alternative on 26. The two sides of each move use opposite value signs. The established all-action solver independently checked 24 of those witnesses. These are training games. [Exact position-value solver](../training/exact_targets.cpp)

## Development games against original Quick

| Player vs original 1,000-visit search | W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- |
| Gen102 | 3–27–18 | 34.4% | 26.0%–42.7% |
| Student trained on the pilot opponent games | 5–30–13 | 41.7% | 33.3%–50.0% |
| Student trained on the new opponent games | 2–36–10 | 41.7% | 34.4%–49.0% |
| Student trained on lower-cost search targets | 4–31–13 | 40.6% | 32.3%–49.0% |

The pilot opponent-trained student scored +7.3 points relative to Gen102 on this set. The paired 95% interval was -3.1 to +17.7 points over 22 rotation and reflection families.

These development games use a separate seed for generated openings and the actual 220 ms browser worker. The published Quick tournament above has a different set of starting positions.

On 763 held roots and 5,777 replies, the student’s mean squared error for child values went from 0.0128 to 0.0105. The mean loss in target value from choosing a suboptimal reply went from 0.0247 to 0.0237.

| One-ply student experiment | Opponent | W–D–L | Games | Score |
| --- | --- | --- | --- | --- |
| Student trained on all legal successor values | Gen102 | 203–106–203 | 512 | 50.0% |
| Student trained on all legal successor values | Original uttt.ai | 153–109–250 | 512 | 40.5% |
| Student value head trained on visit rankings | Gen102 | 191–123–198 | 512 | 49.3% |
| Student value head trained on visit rankings | Original uttt.ai | 181–123–208 | 512 | 47.4% |

The one-ply development games used 256 six-move opening families, played from both seats. Each player evaluated every legal reply. Changes to a value head can affect this benchmark while leaving greedy policy choices identical.

## The networks across generations

| Model vs original uttt.ai | Greedy, two moves | Greedy, six moves | One ply, two moves | One ply, six moves |
| --- | --- | --- | --- | --- |
| Gen101 browser student | 45.1% | 46.7% | 42.8% | 44.6% |
| Gen102 browser student | 47.1% | 48.8% | 42.7% | 38.9% |
| Morning full teacher | 50.5% | 52.1% | 50.8% | 50.0% |
| Afternoon browser finalist | 47.4% | 49.0% | 40.7% | 42.5% |
| Afternoon full teacher finalist | 53.0% | 52.6% | 48.3% | 49.6% |

Scores use the same 720 two-move starts and 256 six-move families in both seats. Greedy selects the most likely legal move. One ply evaluates every legal successor, including exact terminal outcomes. In the two-move suite the change from Gen101 to Gen102 was +2.0 points on the greedy policy and −0.2 points with one ply.

| Timing run | Network | Greedy median, ms | One-ply median, ms | Greedy p90, ms | One-ply p90, ms |
| --- | --- | --- | --- | --- | --- |
| Morning | Browser student | 0.670 | 4.122 | 0.956 | 6.120 |
| Morning | Full teacher | 4.473 | 27.335 | 5.037 | 37.561 |
| Morning | Original uttt.ai | 4.260 | 29.160 | 6.942 | 44.008 |
| Finalists | Browser student | 0.530 | 3.803 | 0.813 | 7.972 |
| Finalists | Full teacher | 4.295 | 27.795 | 6.956 | 41.990 |
| Finalists | Original uttt.ai | 4.170 | 25.028 | 5.205 | 36.546 |

Timings cover a complete move on a warmed, single-threaded browser worker. Each method made 180 timed decisions over the same spread of 60 early, middle and late positions. The one-ply comparison evaluates every legal successor, including exact terminal decisions.

### Replaying complete network-only games in Chromium

| Model | Move rule | Browser W–D–L | Browser score | Native score on same starts | Identical full games | Same final result |
| --- | --- | --- | --- | --- | --- | --- |
| Gen102 | Greedy | 25–71–32 | 47.3% | 47.3% | 128 / 128 | 128 / 128 |
| Gen102 | One ply | 37–21–70 | 37.1% | 37.1% | 123 / 128 | 128 / 128 |
| Full teacher | Greedy | 35–60–33 | 50.8% | 50.8% | 128 / 128 | 128 / 128 |
| Full teacher | One ply | 42–29–57 | 44.1% | 44.1% | 123 / 128 | 128 / 128 |

The browser played both networks from both seats on 64 openings drawn uniformly from the 4,096-family study. All four matchups used the same starts and the website’s fixed-batch original network. Each game ran to completion on a single WebAssembly thread. A separate Python rules implementation checked all 512 outcomes. These games measure runtime agreement on a subset of the opening families used in the larger strength estimate. [Browser network arena](../scripts/browser-policy-arena.mjs)

The public evidence records the sampled opening indices and every browser game. Within each pair the first record puts the project model in X and the second in O. Each consecutive pair of hexadecimal digits in a move record encodes one board-major action.

## When the player can choose a local board

| Legal choices | Deep positions | Gen102 / top move | Full / top move | Original / top move | Solved positions | Gen102 / optimal | Full / optimal | Original / optimal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| One playable board | 3656 | 70.0% | 75.2% | 75.7% | 491 | 90.0% | 92.1% | 90.8% |
| Several playable boards | 392 | 58.2% | 75.3% | 85.7% | 326 | 92.9% | 93.3% | 93.6% |

The original engine searched each deep position for 3,072 visits. The table measures how often a greedy policy picked one of its most-visited moves. The solved column uses independent exact outcomes on positions where the move can change the game result. Positions with only one legal move are excluded. [Boards, search counts and exact outcomes](artifacts/policy-networks-positions-2026-11-26.npz)

To test this difference, two students used identical training boards, targets, parent weights, optimizer seed and 2,600 training updates. One added an objective on the total search probability assigned to each playable local board. Both retained identical value predictions on the parity boards. [Training objective](../training/regret_replay.py)

The extra board-choice objective scored 51.0% against the matched control in 192 six-move opening games at 256 visits (35–126–31, paired 95% interval 48.7%–53.4%).

| Fresh opponent games | Positions | Opening families | Gen102 / top move | Control / top move | Board objective / top move | Original / top move | Board objective minus control, 95% points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| One board | 813 | 19 | 64.2% | 66.4% | 66.4% | 69.6% | -0.7 to +0.8 |
| Several boards | 140 | 18 | 61.4% | 63.6% | 63.6% | 85.0% | +0.0 to +0.0 |

The external check used original-engine searches at 8,192 visits on new browser games and 4,096 visits on new 5 MB games. After checking 358,265 replay rows and their legal successors for overlap, 953 distinct eligible position families remained across the two sources.

A second objective compared boards using the strongest individual move on each board. On the first independent panel, the original greedy model often picked a board containing the best single move even when a different board received more total visits. The two objectives used the same coefficient, training targets, initialization and optimizer seed.

In 192 direct six-move-opening games, the strongest-cell student scored 50.0% against the total-mass student (31–130–31, 95% interval 47.9%–52.1%).

| New six-move openings | Positions | Opening families | Gen102 | Full teacher | Original | Control | Board mass | Strongest cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| One board | 339 | 63 | 68.7% | 82.0% | 85.0% | 71.4% | 71.4% | 71.4% |
| Several boards | 339 | 64 | 60.8% | 74.0% | 84.7% | 61.1% | 61.4% | 62.5% |

After all three controlled students were frozen, 128 independent games started from 64 new six-move opening families. The table measures greedy agreement with fresh 4,096-visit original searches. The single-board comparison matched 328 of 339 multiple-board positions by the number of moves played, in bands of eight. Exact root and immediate-successor overlaps with the training replay were excluded.

### Giving the student more practice choosing a local board

Only 2,316 of 23,351 eligible supplemental training positions (9.9%) offered a choice of local board. A new student drew 31 of its 61 supplemental examples per batch from those positions and 30 from positions with one available board. The control sampled the same positions at their natural frequency. Both used the same 24,567 searched labels, 224 visits per training root, parent weights, optimizer seed, 2,600 updates and 192 examples per batch. The value predictions matched the parent exactly on 396 parity positions. The release finalists had already been selected.

At 256 visits per side the oversampled student scored 49.5% against its uniform control (68–244–72 in 384 games, paired 95% interval 47.7%–51.4%). Both students played each of 192 new six-move starts from both seats.

| Playable boards | Fresh positions | Uniform / top move | Oversampled / top move | Original / top move | Oversampled minus uniform, 95% points |
| --- | --- | --- | --- | --- | --- |
| One board | 348 | 66.4% | 66.4% | 80.5% | -1.4 to +1.4 |
| Several boards | 348 | 63.5% | 64.4% | 80.5% | -0.6 to +2.3 |

After training, 128 new games produced an independent evaluation panel. The original engine searched each selected position for 4,096 visits. The 64 opening families shared none with the previous panel, and selection excluded its positions as well as training positions and their immediate successors. The intervals resample complete opening families.

### Spending deeper training searches on those turns

Each student received 1,024 additional 2,048-visit original-engine labels. One assigned all of them to positions with several playable boards between plies 16 and 47. The control selected positions at random, matching the exact move number of every assigned label. It selected 181 multi-board positions. Both retained the natural training frequencies, the same 24,567 initial 224-visit labels, optimizer seed and 2,600 updates. Each arm used 7,600,160 training visits. The searches for their 160 shared rows supplied the same targets to both students.

The student with deeper labels on multi-board positions scored 49.6% against the equal-work control over 384 independent, seat-swapped six-move games at 256 visits per side (62–257–65, 95% interval 48.7%–50.5%).

| Playable boards | Fresh positions | Random / top move | Targeted / top move | Original / top move | Targeted minus random, 95% points |
| --- | --- | --- | --- | --- | --- |
| One board | 351 | 70.9% | 70.9% | 78.1% | +0.0 to +0.0 |
| Several boards | 351 | 64.4% | 64.4% | 82.9% | +0.0 to +0.0 |

The final panel contains 702 previously unused positions from another 64 six-move opening families. The original engine supplied an independent 4,096-visit reference on each position. The opening families and evaluation positions were disjoint from both earlier board-choice panels.

The deeper search changed the top training move on 174 of 1,024 multi-board positions and 134 of 1,024 randomly assigned positions. After training, the two students chose different greedy moves on 0 of 702 independent evaluation boards.

### Showing the revised targets more often

A final control gave 31 of the 192 examples in every batch to the same 1,024 multi-board positions. One student used their original 224-visit labels and the other used their 2,048-visit labels. Both drew the other 161 examples identically and used the same parent, training seed and 2,600 updates. This isolates the effect of the deeper labels when those examples appear frequently.

The student with deeper labels scored 50.4% against the shallow-label student at 256 visits per side (63–261–60 over 384 six-move games, 95% interval 48.6%–52.1%).

Against the original greedy policy, the shallow-label student scored 45.5% and the deep-label student scored 45.4%. Each played 2,048 games on the same 1,024 new six-move opening families. The paired difference was -0.10 points (95% interval -0.95 to +0.76). Both used one network evaluation per move.

## Cheaper training searches

A 128-visit experiment compared two ways to train from the original engine’s search. The best visit-based control on the calibration data used its raw move frequencies. The other also used each move’s searched value and its one-ply network value. The mixing weights were fixed on separate archived positions. On 1,514 independently held boards, compared with 3,072-visit searches, Jensen–Shannon divergence was 0.0285 for the visit control and 0.0233 for the alternative. Lower values indicate closer agreement. The most-visited action agreed on 83.9% and 85.6%, respectively. The method adapts the training target in [Grill et al., ICML 2020](https://proceedings.mlr.press/v119/grill20a.html).

The divergence fell 18.2% at the same 128 visits. The absolute difference was -0.00518 with a 95% interval of -0.00628 to -0.00406 across 95 independent groups of games. The CPU ONNX implementation reproduced the same top moves in aggregate (85.60%) and a mean divergence of 0.0233124. The resulting training set covers 24,567 additional positions. [Reproducible target generator](../training/regularized_policy.py)

Training the browser student on these targets kept its value predictions fixed. Its greedy score against the original policy went from 47.43% to 46.91% on all two-move starts and from 49.02% to 48.54% on the six-move sample. The gain in label quality did not carry over to greedy playing strength in this run.

## Where to spend additional training visits

David Wu used full searches on a randomly chosen quarter of self-play turns in KataGo and recorded these turns for training. This experiment chose which boards to extend from the first 128 visits and used every board as a training example. [Wu, Accelerating Self-Play Learning in Go, section 3.1](https://arxiv.org/abs/1902.10565)

Every board received a 128-visit search. A rule fitted on separate archived boards used only those shallow results and the network’s predictions to choose one quarter of the boards for an additional independent 384-visit search. For those boards, the two searches were pooled by visit count. The controls spread 224 visits uniformly, or assigned the same number of extra searches at random. All arms used the same policy-target formula.

| Evaluation | Boards | Reference visits | Uniform divergence | Random extra divergence | Targeted extra divergence | 95% targeted minus uniform | Targeted top agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Held general games | 1513 | 3072 | 0.01953 | 0.02148 | 0.01849 | -0.00242 to +0.00037 | 87.2% |
| Fresh games against original | 653 | 8192 | 0.04821 | 0.04926 | 0.04621 | -0.00436 to +0.00021 | 80.6% |

Lower divergence means closer agreement with deep search. The random column is the exact mean over all equally sized random subsets. The evidence file records actual visit counts, opening or source-game groups, top-action agreement and paired intervals against both controls.

### Continuing the same search tree

The next experiment kept the 128-visit tree in memory. Each group contained up to 64 positions. The existing difficulty estimates selected a quarter for another 384 visits. The uniform and random controls used snapshots of the same trees, and an independent check reproduced the unmodified original engine’s counts and values at 128, 224 and 512 visits.

| Dataset | Boards | Targeted / uniform visits | Uniform divergence | Targeted divergence | Divergence reduction | 95% difference | Uniform / targeted top agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| General positions | 1513 | 338,816 / 338,912 | 0.01952 | 0.01651 | 15.4% | -0.00496 to -0.00117 | 87.2% / 87.6% |
| New opponent games | 653 | 146,176 / 146,272 | 0.04819 | 0.04339 | 10.0% | -0.00783 to -0.00209 | 79.8% / 81.5% |

The general boards also informed the preceding diagnostic with restarted searches. The rule was already fixed from separate calibration games. The independently generated opponent games provide the external replication.

The project implementation reproduced the selection of 378 out of 1,513 general boards, mean divergence 0.0165062, and top agreement 87.57%. The evidence file includes the 18 fitted coefficients. [Training search allocation](../training/search_allocation.py)

The training run produced two matched datasets on 24,567 boards. One used exactly 224 visits everywhere. The other spent 128 visits on every board and extended 6,141 of those trees to 512. Its total was 5,502,720 visits against 5,503,008 for uniform search. Computing both controls together took 7,271,616 visits because they could share the beginnings of the searches.

The two students used the same parent, optimizer seed, 2,600 updates, policy-only training and deployment graph. At 256 visits per side, the selected-search student scored 48.8% against the uniformly searched student (18–89–21 in 128 games, 95% interval 46.6%–50.4%).

## Did the one-ply improvement repeat?

| Openings | Games per model | Gen102 vs original | Ranked replies vs original | Change, points | 95% interval, points |
| --- | --- | --- | --- | --- | --- |
| First six-move sample | 512 | 38.9% | 47.4% | +8.5 | +3.1 to +13.9 |
| Every legal two-move start | 1440 | 42.7% | 46.0% | +3.4 | Complete suite |
| Fresh six-move sample | 1024 | 42.0% | 42.4% | +0.3 | -3.4 to +4.1 |

Both networks evaluated every legal reply against the original one-ply network. A complete replay of each move through the shipped game rules verified the scores. Each model played both seats on every start. The six-move samples shared 0 rotation/reflection families. The large first-sample gain narrowed to 0.3 points on 512 fresh opening families, with an interval spanning an even result. The exhaustive two-move comparison improved by 3.4 points.

## Does the fast student benefit from averaging board orientations?

Each network chose the legal move with the highest mean probability across the stated number of views. Four views use all four rotations, and eight include reflections. The opponent always used one greedy pass through the original network. Both seats and every opening match the single-view study. All configurations used zero search simulations.

| Greedy player | All 1,440 two-move games | 512 six-move games | Six-move gain over one view, points (95%) | Browser median ms | Browser p90 ms |
| --- | --- | --- | --- | --- | --- |
| Gen102, one view | 47.1% | 48.8% | Reference | 0.500 | 0.560 |
| Gen102, four views | 48.2% | 50.9% | +2.1 (-1.9 to +6.0) | 1.725 | 1.830 |
| Gen102, eight views | 48.8% | 51.2% | +2.3 (-1.7 to +6.3) | 3.290 | 3.461 |
| Full teacher, one view | 50.5% | 52.1% | Reference | 3.745 | 3.815 |
| Full teacher, four views | 51.5% | 51.4% | -0.7 (-4.7 to +3.1) | 14.650 | 14.746 |

The original single-view player took 3.900 ms per complete greedy move in the same warmed Chromium worker (p90 3.935 ms). All six configurations were timed on the same 60 early, middle and late positions, with 180 counterbalanced measurements each. The multi-view graphs shared the base weights across views. The production game uses a single view per leaf. [Symmetry graphs](../training/symmetry_teacher.py) · [Network arena](../training/policy_arena.py)

### Independent six-move confirmation

| Greedy player vs single-view original | W–D–L | Games | Score | Opening-paired 95% interval |
| --- | --- | --- | --- | --- |
| Gen102, one view | 1834–3982–2376 | 8192 | 46.7% | 45.9% to 47.5% |
| Gen102, eight views | 2010–4042–2140 | 8192 | 49.2% | 48.4% to 50.0% |
| Full teacher, one view | 2127–4207–1858 | 8192 | 51.6% | 50.9% to 52.4% |

Every player faced the same 4,096 additional six-move opening families in both seats. The three graphs and the opening list were fixed before these matches. The change from one student view to eight was +2.51 points (paired 95% interval +1.50 to +3.54). All 24,576 games replayed to the recorded outcomes under the production rules. To reproduce the openings, generate 4,224 six-move families with arena seed 26317950, discard the 768 families from the two earlier six-move samples if encountered, and retain the first 4,096 survivors. The evidence includes the ordered opening digest.

### Equal one-ply lookahead on the same 4,096 families

| Project model | Greedy vs greedy original | One-ply W–D–L | One ply vs one-ply original | One-ply 95% interval |
| --- | --- | --- | --- | --- |
| Gen102 | 46.7% | 2594–1736–3862 | 42.3% | 41.3% to 43.2% |
| Afternoon browser finalist | 46.7% | 2620–1680–3892 | 42.2% | 41.3% to 43.2% |
| Morning full teacher | 51.6% | 3292–1810–3090 | 51.2% | 50.3% to 52.2% |

Each score covers 8,192 games with both seats. At one ply, both networks evaluated every legal successor and chose the move with the best predicted outcome. A terminal reply received its exact value. The greedy comparison selected one legal move from a single policy evaluation on each side.

Against the same original opponents, the afternoon finalist changed the greedy score by -0.04 points (paired 95% interval -0.89 to +0.81) and the one-ply score by -0.02 points (-1.17 to +1.10).

### Distilling eight views into a single evaluation

Two compact students started from the same afternoon parent. Both trained for 2,600 policy-only updates on identical batches. For one 24,567-position source, the target averaged a 224-visit search distribution with the Gen102 network’s own policy. The control used a single Gen102 view and the experimental student used eight. This source supplied 32% of the training mixture. Both resulting graphs use one view per position and occupy 1,476,740 bytes.

| Single-pass player vs greedy original | W–D–L | Games | Score | Paired 95% interval |
| --- | --- | --- | --- | --- |
| Parent | 852–2050–1194 | 4096 | 45.8% | 44.8% to 46.9% |
| Single-view targets | 872–2030–1194 | 4096 | 46.1% | 45.0% to 47.1% |
| Eight-view targets | 875–2032–1189 | 4096 | 46.2% | 45.1% to 47.2% |

These games used 2,048 new six-move opening families, both seats and the same single-pass original opponent. The eight-view training targets changed the score by +0.10 points relative to single-view targets (paired 95% interval -0.16 to +0.34).

In a separate direct match at 256 visits per side, the student trained on eight views scored 50.7% against the single-view control (63–263–58 over 384 games, paired 95% interval 49.5%–51.8%). Both students were research controls outside the release selection.

The numeric archive contains all 73,728 complete move histories from the large independent greedy and one-ply studies, including the matched training controls. Arrays named *moves* use the dimensions opening pair, project seat (X then O), and ply. Actions use board-major indices 0 through 80, with 255 padding after the game. The corresponding *outcomes* arrays record a project win as 1, a draw as 0 and a loss as −1. Each panel also includes the six starting moves. [Download the game records](artifacts/network-games-2026-11-26-afternoon.npz)

## Development matches

| Morning comparator | Challenger | W–D–L | Score | 95% paired interval |
| --- | --- | --- | --- | --- |
| Web | Student trained on the strongest individual move per local board (second client) | 18–63–15 | 51.6% | 46.7%–56.4% |
| Web | Student trained on the strongest individual move per local board (primary client) | 21–84–23 | 49.2% | 44.9%–53.6% |
| Web | Blend of compact checkpoints (primary client) | 17–94–17 | 50.0% | 45.9%–54.2% |
| Web | Blend of compact checkpoints (second client) | 29–51–16 | 56.8% | 51.5%–62.2% |
| Web | Student with extra training on choosing a local board (second client) | 18–69–9 | 54.7% | 50.5%–59.0% |
| Web | Student with extra training on choosing a local board (primary client) | 23–82–23 | 50.0% | 44.8%–55.1% |
| Web | Student with updated weight assignments (second client) | 20–56–20 | 50.0% | 43.7%–56.1% |
| Web | Student with a smaller third residual block (second client) | 12–50–18 | 46.2% | 41.2%–51.3% |
| Web | Student with a smaller third residual block (primary client) | 20–86–22 | 49.2% | 44.9%–53.5% |
| Web | Student trained on paired alternative moves (primary client) | 13–75–8 | 52.6% | 48.0%–57.1% |
| 5 MB | 5 MB teacher trained with six-bit rounding | 20–87–21 | 49.6% | 44.6%–54.4% |
| 5 MB | Six-bit teacher | 19–91–18 | 50.4% | 45.3%–55.6% |
| 5 MB | 5 MB trained on its own games with four policy-scale tensors | 22–81–25 | 48.8% | 44.8%–52.8% |
| 5 MB | Six-bit teacher with retrained scales | 19–86–23 | 48.4% | 43.7%–53.1% |
| 5 MB | Full teacher trained on its own original-engine games | 21–90–17 | 51.6% | 47.2%–56.1% |
| 20 MB | Full teacher trained on its own original-engine games | 25–84–19 | 52.3% | 47.3%–57.4% |
| 25 MB | Full teacher trained on its own original-engine games | 24–84–20 | 51.6% | 46.5%–56.6% |
| 25 MB | Full teacher trained on every legal reply | 19–88–21 | 49.2% | 46.0%–52.4% |
| Web | Student trained on rotation-averaged labels (primary client) | 14–69–13 | 50.5% | 44.3%–56.4% |
| Web | Opponent-trained student with the transferred reply-ranking head (second client) | 15–65–16 | 49.5% | 43.8%–55.1% |
| Web | Opponent-trained student with the transferred reply-ranking head (primary client) | 22–91–15 | 52.7% | 48.5%–56.9% |
| Web | Student trained on 8,192-visit pilot opponent positions (primary client) | 24–90–14 | 53.9% | 50.0%–57.8% |
| Web | Student trained on 8,192-visit pilot opponent positions (second client) | 20–68–8 | 56.2% | 51.6%–61.0% |
| Web | Student trained on the two new opponent datasets (second client) | 26–59–11 | 57.8% | 52.1%–63.5% |
| Web | Additional trained candidate (primary client) | 26–85–17 | 53.5% | 48.4%–58.6% |
| 5 MB | Layer-wise blend of full teachers | 17–92–19 | 49.2% | 45.6%–52.8% |
| 20 MB | Layer-wise blend of full teachers | 23–86–19 | 51.6% | 47.6%–55.8% |
| 25 MB | Layer-wise blend of full teachers | 22–87–19 | 51.2% | 47.3%–55.1% |
| 5 MB | Learned corrections for local and global threats | 16–95–17 | 49.6% | 47.6%–51.6% |
| Web | Learned corrections for local and global threats (second client) | 14–50–16 | 48.8% | 43.1%–53.8% |
| 25 MB | Learned corrections for local and global threats | 19–91–18 | 50.4% | 48.8%–52.0% |
| 512 KB | Learned corrections for local and global threats | 29–75–24 | 52.0% | 48.8%–55.1% |
| Web | Student value head trained on visit rankings (second client) | 14–59–23 | 45.3% | 39.9%–50.5% |
| Web | Student value head trained on visit rankings (primary client) | 19–86–23 | 48.4% | 43.5%–53.8% |
| Web | Student trained with extra visits on selected positions (second client) | 21–59–16 | 52.6% | 46.1%–58.9% |
| Web | Student trained with extra visits on selected positions (primary client) | 22–87–19 | 51.2% | 46.4%–55.9% |
| Web | Student trained to repair supported errors in its own played moves (second client) | 23–60–13 | 55.2% | 50.0%–60.6% |
| Web | Student trained to repair supported errors in its own played moves (primary client) | 21–95–12 | 53.5% | 49.2%–57.8% |
| 5 MB | Own-game full teacher with the previous value layers | 20–88–20 | 50.0% | 45.2%–55.1% |
| 20 MB | Own-game full teacher with the previous value layers | 21–92–15 | 52.3% | 47.6%–57.0% |
| 25 MB | Own-game full teacher with the previous value layers | 21–89–18 | 51.2% | 46.2%–56.2% |
| Web | Student trained on lower-cost search targets (second client) | 22–58–16 | 53.1% | 47.4%–58.9% |
| Web | Student trained on lower-cost search targets (primary client) | 21–88–19 | 50.8% | 46.4%–55.2% |
| 512 KB | Tiny trained with exact integer rounding | 28–72–28 | 50.0% | 44.9%–55.2% |
| 512 KB | Refitted Tiny INT8 scales | 32–84–12 | 57.8% | 53.6%–61.9% |
| Web | Student trained on all legal successor values (second client) | 18–60–18 | 50.0% | 44.3%–55.7% |
| Web | Student trained on all legal successor values (primary client) | 20–87–21 | 49.6% | 45.2%–54.5% |

These matches guided candidate selection. The fresh confirmation seeds above determine what ships.

## Which investments paid off

The clearest deployment gain this morning came from doing less work per neural evaluation. With Gen101’s weights fixed, the legal-cell decoder raised median visits from 277 to 384 on the primary client and from 94 to 165 on the second. The morning 5 MB model also qualified by compressing a stronger full teacher into 4,113,260 bytes.

The afternoon browser finalist scored 51.6% on the primary client and 51.6% on the second. Those games and the published Quick comparison determined the decision to keep Gen102.

The 5 MB gain came from compressing the teacher after training it on its own games against the original engine. The six-bit export scored 53.8% against the morning six-bit file over 384 games. The same training at full precision scored 50.9% against the full teacher, with a 95% interval of 48.1%–53.7%.

The new Tiny model won its fresh same-size comparison with 56.5% over 384 games. Its scale update preserved the deployed search protocol and all integer weight assignments.

For training, adding searched action values and one-ply evaluations to the 128-visit targets reduced their error against deep search. The one-ply value-head experiment yielded an apparent 8.5-point gain that shrank to 0.3 points on a fresh set of six-move openings.

Continuing selected search trees improved the accuracy of the supervision. At identical training cost, the resulting student scored 48.8% in direct play against its uniform-search control. That experiment leaves its deployment benefit unresolved.

For the next round, generate more independent games against Quick and Strong at their published visit settings. Use those games to collect more distinct middle-game positions where several local boards are playable, then compare against equally many game-phase-matched positions from general play at the same search depth. Keep complete opening families separate through training and evaluation. Compare both the greedy decisions and actual Hard games on fresh seeds before adopting the change.

The 29 tests across the six affected training modules passed together, including exhaustive game-rule checks of the position solver. The evidence records the hashes of the tested sources.

The complete training suite ran 122 tests across 34 modules with 0 failures, 0 errors and 0 skipped. The source snapshot digest is included in the evidence.

The application build, unit tests, worker tests, production checks and desktop and mobile browser tests passed against the selected browser model.

The separate study measures teacher and student policy strength, solved-position decisions and warmed browser latency. [Network-only study](policy-networks-2026-11-26.md) · [Afternoon scorecards and diagnostics](artifacts/model-strength-2026-11-26-afternoon.json)
