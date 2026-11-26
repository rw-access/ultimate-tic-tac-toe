# Student and teacher against the original uttt.ai network

Gen102 made a greedy policy evaluation in **0.565 ms** on a single browser worker, 7.8 times as fast as the original network. Its score against that network was **47.1%** across all two-move starts and **48.8%** on a separate six-move sample. The full teacher scored **50.5%** and **52.1%** with policy alone. Comparing the predicted value of every legal reply favored the original over the student and left the teacher close to even.

## Paired games

Both players used the same move rule. Greedy policy selected the legal move with the highest logit in one pass. At one ply each player tried every legal move, took the exact outcome for a terminal position and otherwise negated its value prediction for the opponent to move. Ties went to the lowest board-major move index. Each network saw one orientation per position. These games used zero MCTS simulations, no opening book and no endgame solver.

| Project model | Move rule | Two-move W–D–L | Score | Six-move W–D–L | Score | Six-move 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| Gen102 student | Greedy policy | 316–724–400 | 47.1% | 121–258–133 | 48.8% | 46.0%–51.6% |
| Full teacher | Greedy policy | 357–740–343 | 50.5% | 132–269–111 | 52.1% | 49.3%–54.9% |
| Gen102 student | One-ply value | 466–297–677 | 42.7% | 144–110–258 | 38.9% | 35.0%–42.8% |
| Full teacher | One-ply value | 584–296–560 | 50.8% | 198–116–198 | 50.0% | 46.3%–53.7% |

The first suite exhausts all 720 legal ordered two-move starts and both seats, or 1,440 games per matchup. Its scores are exact for these deterministic starts. The second suite draws 256 different six-move opening families, counting rotations and reflections together, for 512 games per matchup. Its intervals resample opening pairs. All eight matchups used the same starts within each suite. A win earns one point and a draw earns half. The shipped game rules independently replayed all 7,808 games.

## Move choices on independently evaluated positions

The deep-search panel contains 4,096 distinct position families. The original engine used 3,072 root visits per position. The table counts choices tied for the most visits and gives the mean visit share of the chosen move. The original network supplies the priors and evaluations for this reference. Agreement with it therefore measures how closely each player imitates that particular search.

The exact panel provides an independent reference. A solver classified every legal action as a win, draw or loss at 2,048 late positions. On 817 of them, the choice could change that outcome. The table counts any move that preserves the best achievable result on those 817 positions. The other 1,231 have equal outcomes for all legal choices. Half the sampled positions were inside the browser’s 14-playable-square solver horizon, and half were outside it. A network received no solver output when choosing.

| Model | Move rule | Most-visited move | Chosen visit share | Optimal solved move |
| --- | --- | --- | --- | --- |
| Gen102 student | Greedy | 2,834/4,096 (69.2%) | 56.9% | 745/817 (91.2%) |
| Gen102 student | One ply | 2,283/4,096 (55.7%) | 49.9% | 730/817 (89.4%) |
| Full teacher | Greedy | 3,092/4,096 (75.5%) | 59.4% | 756/817 (92.5%) |
| Full teacher | One ply | 2,468/4,096 (60.3%) | 52.2% | 751/817 (91.9%) |
| uttt.ai Stage 2 | Greedy | 3,153/4,096 (77.0%) | 59.5% | 751/817 (91.9%) |
| uttt.ai Stage 2 | One ply | 2,469/4,096 (60.3%) | 52.6% | 751/817 (91.9%) |

On the 817 positions with a consequential choice, the teacher’s greedy policy produced a better proved result than the original policy on 11 positions and a worse one on six. For the student the counts were 14 and 20. These are small, position-specific differences. The game results and the solved positions together support approximate parity for the teacher and a measurable gap for the student’s one-ply values.

Three solved positions show how those differences look on the board. Each has exactly one winning move. All three are outside the browser solver horizon. The row numbers are zero-based indices in the downloadable exact-position array. Each player used its greedy policy.

| Exact row | Playable squares | Unique winning move | Gen102 choice | Teacher choice | Original choice |
| --- | --- | --- | --- | --- | --- |
| 1218 | 22 | Top left cell of center board | Loss | Win | Loss |
| 1504 | 20 | Bottom right cell of top right board | Win | Loss | Loss |
| 2014 | 15 | Middle left cell of center board | Draw | Draw | Win |

## Speed in a browser and on a native CPU

The browser measurements used a warmed Chromium worker and single-thread WebAssembly. Native measurements used a single ONNX Runtime CPU thread on the same machine. There were 60 board positions, 20 each from early, middle and late play. The forward-pass benchmark used pre-encoded inputs, batch size one, counterbalanced model order and 432 timed passes per configuration. Startup and download time are excluded. The policy-only graphs remove the value branch. Their policy outputs matched the shipping graphs exactly on all 60 positions in both runtimes.

| Model | Browser policy only, median ms | Browser both heads, median ms | Browser policy only, p90 ms | Native policy only, median ms | Native both heads, median ms |
| --- | --- | --- | --- | --- | --- |
| Gen102 student | 0.565 | 0.630 | 0.820 | 0.283 | 0.318 |
| Full teacher | 4.285 | 4.432 | 6.540 | 0.526 | 0.560 |
| uttt.ai Stage 2 | 4.380 | 4.605 | 5.240 | 0.507 | 0.534 |

The complete-move benchmark includes legal move generation, board updates at one ply, input encoding, inference, move selection and disposal of tensors. It uses each unmodified shipping graph with both outputs, 180 timed choices per mode, the same 60 positions and batch size one for all networks. Every legal nonterminal successor receives one evaluation. The following numbers are wall-clock milliseconds on this browser.

| Model | Greedy median | Greedy p90 | One-ply median | One-ply p90 | Median one-ply evaluations |
| --- | --- | --- | --- | --- | --- |
| Gen102 student | 0.670 | 0.956 | 4.122 | 6.120 | 6 |
| Full teacher | 4.473 | 5.037 | 27.335 | 37.561 | 6 |
| uttt.ai Stage 2 | 4.260 | 6.942 | 29.160 | 44.008 | 6 |

All three greedy players and the student at one ply made the same moves across browser and native inference on all 60 timing boards. The teacher and original each chose a different move at one ply on the same position. Their browser values for the competing moves were within 0.00000024 of each other near a predicted loss. The match results above use native inference throughout.

The teacher and original were close in browser forward-pass time. Native CPU ordering reversed. Device, runtime and the number of legal replies affect the timing. The production Hard deadline comparison is a separate measurement. [Hard match report](model-strength-2026-11-26-morning.md)

## Reproduction

The game harness accepts two ONNX files, an opening depth of two or six and either move rule. Both modes used seed `26316756`. The original game runs used its batch-enabled Stage 2 graph. All 59 operators and 48 initializers matched the fixed-batch website graph, and both output heads agreed numerically on fixtures covering both players. The single-position speed benchmark used the original website graph itself. [Game harness](../training/policy_arena.py) · [Original uttt.ai code and model](https://github.com/arnowaczynski/utttai)

The evidence includes model hashes, raw match hashes, seat breakdowns, paired uncertainty, latency distributions and source panel hashes. The numeric position archive stores project-format states, integer visit counts, exact action outcomes, and the moves each network chose under each rule. For exact outcomes, −2 marks an illegal action, −1 a loss, 0 a draw and 1 a win from the current player’s perspective. [Evidence](artifacts/policy-networks-2026-11-26.json) · [Positions and decisions](artifacts/policy-networks-positions-2026-11-26.npz)
