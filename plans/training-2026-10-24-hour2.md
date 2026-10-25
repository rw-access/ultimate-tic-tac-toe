# Additional one-core training hour — 2026-10-24

The user explicitly authorized **60 more minutes**, initially on one core.
Start: 22:51:07 UTC. Hard deadline: 23:51:07 UTC. Reserve final validation from
23:48:37 UTC. All training and evaluation commands remain pinned to CPU 0 with
numerical and ONNX threads limited to one. Later the user authorized up to
1.5 cores: the main run stays on CPU 0, while one auxiliary job uses CPU 1
with a verified cgroup quota of 50% (`cpu.max=50000 100000`).

Continue from generation 5 (SHA-256
`cdc02f6847c069ad96fcdb8add2831683e4997a670842f4d09a1fc3a986d5be9`).
The current deployed model remains unchanged unless new paired matches establish
an improvement. Keep the original hour’s provenance and checkpoint files.

Round 8 has started with 600 games, 160 simulations per move, 24 concurrent
games, no heuristic blend, and generation 5 guiding both players. Independent
inspections will check the training plateau and search/exploration behavior
before choosing the next bounded experiment. Reuse the existing rules engine,
small ONNX architecture, and established deployment path.

## Progress

- Rechecking R7 against R5 over 512 games scored 56.45%, paired 95% interval
  [52.59%, 60.30%]. Its original 32-game screen had incorrectly filtered out
  that improvement. The fixed-seed 50-game heuristic guard scored 64%, so R7
  became the local incumbent.
- R8 trained from R7 using the already-generated 600-game batch. It scored
  59.96% against R7 over 512 games, interval [55.93%, 63.99%]. Its 50-game
  heuristic score was 60%, narrowly missing the initial 61% guard threshold.
- A larger guard check at seed 3061024 used 200 games per model: R7 scored
  57.75%, R8 scored 55.5%. The 2.25-point difference passed the agreed 3-point
  guard, so R8 was promoted. These aggregate reports do not support a paired
  difference confidence interval; the direct 512-game arena supplies the
  evidence of improvement over R7.
- Wider screens now use 32 pairs and a 50% threshold; promising candidates get
  two independent 128-pair confirmations, pooled before promotion. Training
  uses learning rate 0.0001, value-loss weight 0.25, up to eight epochs with
  held-out selection/patience 3, and a 100-second limit. New batches use pure
  model-guided search at 256 simulations and 24 concurrent games.
- R9 is generating 1,500 games from R7; after generation completes, training
  will start from the now-stronger R8. The data sidecar preserves its original
  guiding model hash.
- The user additionally requested a throughput subagent to investigate
  200+ games/sec. Its profiling/prototypes are bounded to CPU 0 and must
  distinguish equal-search speedups from cheaper search or raw-policy games.

Further candidate decisions, independent final evaluation, and validation
results will be filled in at completion. Local records remain in
`training/runs/session-20261024T214942/`; the phase-one manifest was archived as
`session.before-hour2.json`.

## Subsequent decisions and performance investigation

- R9 passed 512 direct games against R8 with 56.05% score, paired 95% interval
  [52.08%, 60.03%], and the fixed development heuristic guard at 62%.
- R10 failed its direct confirmation and was rejected.
- R11 scored 56.15% against R9 over 512 games, interval [52.07%, 60.24%].
  Its 50-game development heuristic score was 58%, versus R9's 62%, so promotion
  waited for a larger check. At seed 3061024 over 200 games each, R9 scored
  54.5% and R11 scored 58%. R11 was promoted after passing this larger guard;
  this development check is separate from the final reporting-only evaluation.
- R12's 900 games were generated with R9, then training resumed from R11.
  The superseded R12/R9 weight update was stopped, and data/model provenance
  keeps these two attempts separate. R14 is prefetched as 500 games at the same
  256-simulation quality on the auxiliary half core, guided by R11.
- A direct-policy generator reached roughly 373 games/sec in a controlled
  single-core benchmark and produced byte-identical data to the existing
  one-simulation generator. This is substantially cheaper search, not a
  speedup to the 256-simulation training target. A controlled 20% direct-game
  supplement versus deep-only replay retained the parent weights in both arms
  after two epochs; there was no evidence to adopt it. The main run continues
  deep search. Native inference is already used; a Rust rewrite was deferred
  because weight inference and the number of searched positions dominate.
- The browser search now optionally proves immediate global wins at interior
  leaves and filters root moves that permit an immediate global loss when a
  safe move exists. Medium/Hard enable this; offline training/evaluation keeps
  the old default for comparable model measurements. Eleven AI tests include
  the adversarial low-prior winning-reply regression. Current code passed all
  25 Node tests, all 28 browser tests, build, and production model/layout checks.
- The reported mobile board shrink was fixed to use available width and allow
  a little vertical scrolling. At 390px viewport width the board is 362px wide.
  The existing 280-feature model already explicitly encodes own/opponent/drawn
  macro-board outcomes and the forced destination, in addition to local marks.

## Final result

Round 14 was retained. It scored 55.76% against R11 over 512 games, paired
95% interval [51.62%, 59.91%], and passed the fixed development heuristic guard
at 72%. R12 and R13 failed confirmation; R15 and R16 failed screening. The final
300-game R16 batch used R14 and the remaining idle main core; its bounded
weight update finished with time reserved for release validation.

The additional hour produced 7,000 new main self-play games, bringing the total
to 17,469 across 16 rounds, plus the original 1,000 heuristic-teacher games.
The separate 1,000-game direct-policy experiment is excluded and was not used
by the retained model. Auxiliary prefetches each produced 500 full-quality
256-simulation games, using about 329 and 328 wall seconds at half a core.

Final reporting-only evaluation (seed 4061024), equal 64-simulation search:
218 wins, 56 draws, 126 losses against the heuristic in 400 games, 61.5% score.
Raw policy scored 96% against random. A fresh swapped-seat arena against the
previously deployed R5 (seed 34102499) scored 57.03% over 256 games, paired
95% interval [51.41%, 62.65%]. These final seeds did not select champions.
The browser additionally enables exact tactical checks on Medium/Hard, so this
model-only comparison does not measure that search improvement.

Retained model SHA-256:
`b5bbe2fe6020a0d54af624d0ca41e5f5dde618d4a4df35f5283a063d41622fff`.
Model size and architecture remain 462,865 bytes and 115,410 parameters.
The exact weights passed all 25 Node tests, 12-case native parity, browser
parity on desktop/mobile, and production model play beneath a path prefix.
The same code had already passed the full 28-browser-test suite and build.
Final publication and live validation use the established static-asset path;
host-specific records are maintained outside this repository.

Final live desktop/mobile model play and strict HTTPS artifact checks passed.
The 390px phone board measures 362px and allows vertical scrolling to controls.
All training/auxiliary jobs ended before the hard deadline. Original phase-one
trial labels are preserved in the generation summary; the phase-one plan records
later gentle-retry/larger-arena promotions, while the extension's accepted
checkpoint list records its subsequent promotion history.
