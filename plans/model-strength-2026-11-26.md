# Gen101 and the November model matches

Gen101 scored **67.2% against Gen69 in the production Hard worker**. It fits in 1,480,935 bytes, 11,825 bytes smaller than the deployed Gen69, with the same 384-simulation, 220 ms search budget and the same bounded endgame solver. The 512 KB student scored **94.5%** against its predecessor. Scores award one point for a win and half a point for a draw.

## Frozen finalists

Each model played 256 games against its predecessor and 192 apiece against the original uttt.ai Stage 2 weights and engine. These matches gave both sides 256 visits per move. In the weights column, both players used identical project search. The engine column uses the upstream NMCTS implementation for the opponent. Parentheses give 95% bootstrap intervals clustered by ordered opening. [Full scorecards, model hashes, opponents and training counts](artifacts/model-strength-2026-11-26.json)

| Size class | ONNX bytes | Previous W / D / L | Previous score (95% interval) | uttt.ai weights | uttt.ai engine (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 KB | 477,280 | 234 / 16 / 6 | 94.5% (91.5–97.1%) | 43.2% | 39.6% (35.5–43.8%) |
| Web | 1,480,935 | 122 / 104 / 30 | 68.0% (64.2–71.8%) | 49.0% | 47.1% (42.9–51.5%) |
| 5 MB | 4,209,067 | 218 / 33 / 5 | 91.6% (88.9–94.1%) | 49.5% | 55.2% (51.4–59.0%) |
| 20 MB | 10,014,318 | 165 / 78 / 13 | 79.7% (75.8–83.5%) | 51.8% | 57.3% (53.3–61.2%) |
| 25 MB | 19,989,590 | 180 / 63 / 13 | 82.6% (79.1–86.1%) | 52.3% | 57.3% (53.3–61.2%) |
| 50 MB | 19,989,590 | 134 / 108 / 14 | 73.4% (69.6–77.5%) | 52.3% | 57.3% (53.3–61.2%) |
| 75 MB | 19,989,590 | 133 / 98 / 25 | 71.1% (66.6–75.4%) | 52.3% | 57.3% (53.3–61.2%) |

The 25 MB, 50 MB and 75 MB tiers selected the same full-precision teacher. Each has a distinct predecessor match. Their uttt.ai columns reference the same 192-game matches.

The web predecessor was the publicly deployed Gen69. Its 1,492,760 bytes matched the weights restored by commit `c928e39` (SHA-256 `b6f888ca2a7c987e3366ba2272e01e90f01e7799d74588ed1921c9a2e473485d`). The 50 MB and 75 MB predecessors were V26 and Grandmaster. In additional matches, the 50 MB finalist scored 79.2% against Archived 50 MB V54 (123 / 58 / 11 in 192 games) and the 75 MB finalist scored 75.5% against 75 MB Mega Grandmaster (122 / 46 / 24 in 192 games).

An engine control gave both sides the original Stage 2 weights and 256 visits. Project search scored 58.3% against upstream NMCTS in 96 games (26 wins, 60 draws, 10 losses). The engine contributes to the measured score. In a separate comparison of learned weights with identical project search, the initial fine-tuned teacher scored 53.3% in 256 games, with a lower confidence bound of 49.995%.

## Production browser and published difficulty levels

| Match | Games | Wins / draws / losses | Score | Opening-cluster 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Gen101 vs deployed Gen69, production Hard | 192 | 85 / 88 / 19 | 67.2% | 62.9–71.4% |
| Gen101 vs Gen69, final release build on the original machine | 192 | 85 / 87 / 20 | 66.9% | 62.2%–71.6% |
| Gen101 vs Gen69, final release build on a second machine | 128 | 43 / 67 / 18 | 59.8% | 53.1%–66.2% |
| Gen101 vs Gen69, earlier independent Hard match | 192 | 85 / 88 / 19 | 67.2% | 62.6–71.6% |
| Gen101 vs archived Gen100, 256 visits each | 256 | 145 / 95 / 16 | 75.2% | 71.8–78.5% |
| Gen101 production Hard vs uttt.ai Quick (1,000 visits) | 100 | 6 / 66 / 28 | 39.0% | Complete published suite |
| Full teacher, native 384 visits vs uttt.ai Quick (1,000 visits) | 100 | 15 / 68 / 17 | 49.0% | Complete published suite |
| Full teacher, native 384 visits vs uttt.ai Strong (10,000 visits) | 32 | 0 / 10 / 22 | 15.6% | 8.3–23.4% |

Both Hard workers had the same caps. The two additional matches used the same exact release build on different machines, with worker SHA-256 `d7805afb32f0d0eea9a1d9bcc356dc36f91c2c35a3976c6aa0edfc0c50bab485` for both seats. Across scored turns in the final match, the median was 304 completed simulations for Gen101 and 384 for Gen69. On the second machine the new model completed a median of 99 visits versus 211 for Gen69. Model latency and early exact solutions affect how many visits finish. Across 128 fixed-visit games against each earlier web model, Gen101 scored 72.7% against Gen66, 74.6% against Gen88 and 91.4% against the archived Mega student.

The Quick comparisons use all 50 published tournament positions in both colors. The Strong comparison uses 16 independently generated opening pairs. The full teacher used native search with a 384-visit cap and the existing endgame solver. Gen101 used the actual Chromium Hard worker and its time limit in the Quick match. [Upstream tournament openings](https://github.com/arnowaczynski/utttai/blob/622dae66b18198f6f12aef0a0bccd5bd40de2d8e/utttpy/selfplay/evaluation_uttt_states.py)

## Trained models

Gen101 retains dense gating and value layers, and factorizes four spatial convolutions from the published Stage 2 network. After 8,500 updates on backup targets and solved endgames, the student was packed to 4-bit codebooks. Another 7,800 distillation updates refined the centroids. A board-major matrix multiplication graph made the result practical in the production worker. The trained float model had 2,250,498 parameters. The shipped graph reproduced the source graph's top legal move on all 138 comparison positions. [Network](../training/stage2_student.py) · [Codebook refinement](../training/refine_packed.py) · [Web graph](../training/fast_stage2_graph.py)

The 512 KB student learned board-line and directed-move features from scratch. A calibration fit used 8,000 positions and validated the value adjustment on another 4,000 from separate shards, with symmetry-equivalent positions deduplicated. Before the final tanh, the accepted value scale was 1.257161 and the bias was −0.026042. The policy scale stayed at 1. The model uses 8-bit packed weights. The 5 MB export uses refined 4-bit codebooks, the 20 MB export uses half-precision weights, and the full teacher uses float32. [Line model](../training/line_student.py) · [Calibration](../training/calibrate_outputs.py) · [Downloads and checksums](https://github.com/rw-access/ultimate-tic-tac-toe/releases/tag/models-2026-11-26)

## How the matches were run

- The predecessor and published-weights comparisons used fresh search roots on every turn, 256 visits, identical project search, and no tactical proof or wall-clock deadline. Each two-ply opening was played from both seats.
- The engine comparison used the original uttt.ai Stage 2 NMCTS with exploration 2.0 and tree reuse. Project search used tactical proof and subtree reuse. Both sides had 256 visits. The published ONNX hash is `81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2`. Its batch-enabled export preserved initializer bytes and matched the original on 12 parity cases. [Upstream search implementation](https://github.com/arnowaczynski/utttai/blob/622dae66b18198f6f12aef0a0bccd5bd40de2d8e/website/src/selfplay/NeuralMonteCarloTreeSearch.js)
- The 64-game selection matches used seed `2026112601`. Candidates within five score points of the best against their predecessor were ranked by their result against the public weights. Every selected file fit within its predecessor’s actual byte count.
- Browser selection also ran the leading candidates in the production worker. A challenger needed a 95% lower bound above 50% in a direct 128-game match to displace the provisional winner. The final refined student scored 49.2% against Gen101 and the line model scored 47.7%.
- After freezing the weights, the final matches used seed `2026112802` against predecessors, `2026112804` against the public weights, `2026112872` against the published engine, and `2026112881` for the production Hard match. Each pair had an independently derived opening seed. The promotion match required a lower 95% bound above 50%, plus passing parity, size, build and gameplay checks.

## Repeating the browser comparison

From the release checkout, install the Node dependencies and Chromium as described in the project README. Save the old web model separately and play both seats through the compiled worker. Browser timing depends on the client.

```sh
mkdir -p training/runs/gen101-check
git show c928e39:public/models/policy-value.onnx > training/runs/gen101-check/gen69.onnx
npm run build
node --experimental-strip-types scripts/browser-arena.ts \
  --challenger public/models/policy-value.onnx \
  --champion training/runs/gen101-check/gen69.onnx \
  --pairs 96 --seed 2026112881 --opening-schedule independent \
  --output training/runs/gen101-check/hard.json
```

## Training inputs

The public Stage 2 archive contains 8,023,020 positions evaluated at about 10,000 simulations each. Its SHA-256 is `533a885a0b1f5ea2433a9de73c41d839b282c2abbc5ec69ae0246119eb9eb250`. The initial paired training experiment uses 1.2 million depth-stratified positions and 100,874 archived replay positions. Additional targets include a one-ply backup over every legal reply at 480,000 positions, all-action exact outcomes for 209,478 late positions, 1,024-visit searches on sampled positions, and fresh self-play. The published archive records moves selected by the previous Stage 1 network during Stage 2 data generation. The published dataset and engine are described in the [uttt.ai dataset documentation](https://github.com/arnowaczynski/utttai/blob/main/datasets/README.md) and [pinned search source](https://github.com/arnowaczynski/utttai/blob/622dae66b18198f6f12aef0a0bccd5bd40de2d8e/website/src/selfplay/NeuralMonteCarloTreeSearch.js).

The completed search corpus contains 65,536 distinct sampled positions in 256 verified shards. The two networks each handled 32,768 positions. Across the corpus, 702 immediate wins took no search and the remaining roots used 66,390,016 visits. Fresh self-play added 22,995 saved positions from 1,536 completed games, plus 3,828 positions recovered from 256 earlier games. The final combined corpus contains 92,359 training rows.
