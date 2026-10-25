# Offline policy/value model

The game engine, self-play search, feature encoder, evaluation, and parity
fixtures live in TypeScript. Python consumes the resulting examples, trains a
small network, and exports ONNX. This keeps game rules in one implementation.

Install the pinned CPU dependencies into the project virtual environment:

```sh
uv venv .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
uv pip install --python .venv/bin/python -r training/requirements.txt
```

The TypeScript tools use `onnxruntime-node` for native inference inside the
self-play process. Python only handles dataset files and model exports between
rounds; there is no per-position subprocess bridge. Install the Node dependencies
with `npm ci` before running the tools.

Generate TypeScript self-play positions, train, then evaluate and make parity
fixtures with the TypeScript tools:

```sh
npm run ai:generate -- --games 1000 --simulations 24 --seed 20261024
.venv/bin/python -m training.pipeline train \
  --data training/data/baseline.jsonl --epochs 12 --batch-size 512
npm run ai:evaluate -- --model public/models/policy-value.onnx \
  --games-per-seat 10 --simulations 32 --metadata public/models/metadata.json
npm run ai:parity -- --model public/models/policy-value.onnx
```

For continued training, prefer a separately generated holdout that is never
included in replay or any earlier training run:

```sh
.venv/bin/python -m training.pipeline train \
  --data training/data/replay.jsonl \
  --validation-data training/data/frozen-validation.jsonl \
  --initial-model public/models/policy-value.onnx \
  --hidden-widths 512,256 \
  --augmentation-mode random --threads 1 \
  --output training/runs/candidate.onnx \
  --metadata training/runs/candidate.metadata.json
```

With `--validation-data`, every game in `--data` is used for training and the
separate file is used only for validation. The trainer rejects exact overlap by
hashing each game's ordered feature states and value labels, independent of its
game index and policy target. Keep the holdout file out of every training and
replay dataset. Without this option, the trainer retains its seeded game-level
split within `--data`. Metadata records both file hashes and validation mode,
plus policy-target entropy/KL and value MSE baselines (zero, validation mean,
and training mean) for interpreting the learned losses.

The default hidden widths remain `256,128`. `--hidden-widths H1,H2` permits a
capacity experiment, for example `--hidden-widths 512,256`. When warm-starting,
the trainer reads the parent widths from its ONNX tensors and allows equal or
larger widths only. Wider checkpoints use seeded Net2Wider initialization:
duplicate units split their outgoing weights, with small zero-sum perturbations
to let copies learn separately while initially preserving parent predictions.
The run metadata records parent/target widths, method, seed, and perturbation
scale. Exact-width warm starts still load the original weights unchanged.
For paired architecture experiments, set the same `--shuffle-seed` in both
runs so different parameter counts do not change the minibatch order. If omitted,
the trainer retains its prior global-RNG shuffle behavior.

Browser model artifacts have a hard maximum size of 1,500,000 bytes. The
trainer checks a conservative architecture-based estimate before fitting and
the exact ONNX file size after export; the production build and release process
enforce the same limit before a model can be released.

For datasets with different numbers of saved positions, `--max-updates N`
matches the optimizer-step budget. It can stop partway through the final epoch,
which is still validated before selecting and exporting the checkpoint. Metadata
records executed steps, steps for the selected checkpoint, and the stop reason.

For another round, generate examples with the learned policy/value network:

```sh
npm run ai:generate -- --games 1000 --simulations 128 \
  --model public/models/policy-value.onnx --parallel-games 16 \
  --output training/data/selfplay-v2.jsonl
.venv/bin/python -m training.pipeline train \
  --data training/data/selfplay-v2.jsonl \
  --initial-model public/models/policy-value.onnx \
  --augmentation-mode random --threads 1 --patience 3 \
  --output training/runs/v2.onnx --metadata training/runs/v2.metadata.json
npm run ai:evaluate -- --model training/runs/v2.onnx \
  --games-per-seat 25 --simulations 128 --metadata training/runs/v2.metadata.json
```

Keep future candidate models separate until evaluation supports replacing the
bundled checkpoint. The bundled R16 checkpoint is
`public/models/policy-value.onnx`; its checksum and evaluation are recorded in
`public/models/metadata.json` and `plans/training-2026-10-25-12h.md`.
Intermediate replay data and per-round run artifacts are kept out of the source
repository. Regenerate `training/parity.json` after replacing the bundled
checkpoint.
`--simulations` counts explored paths per move, rather than a fixed depth.
Increasing it improves the search budget and also increases generation time.

`--initial-model` warm-starts from an ONNX checkpoint. The trainer also evaluates
that parent before updating it and keeps the parent weights if no epoch improves
the held-out loss. `--patience` stops after consecutive epochs without improvement;
`--max-seconds` is an epoch-boundary time limit. `--augmentation-mode random`
applies a random symmetry per example per epoch without storing eight copies of
the replay data. Keep complete games together when combining replay datasets.

Use a paired match before replacing an incumbent:

```sh
node --experimental-strip-types scripts/ai.ts arena \
  --challenger training/runs/v2.onnx --champion public/models/policy-value.onnx \
  --pairs 64 --simulations 64 --seed 9102401 \
  --output training/runs/v2-arena.json
```

Each pair reuses the same two-move opening and swaps which model plays X/O.
The report includes win/draw/loss totals, per-pair scores, model hashes, and an
approximate 95% interval over pair scores. Use fresh openings for confirmation
after screening candidates. The training-only `generate --teacher-mix 0.2`
blends 20% of the heuristic evaluator into model-guided search; it records this
setting in provenance and does not change browser inference.

For a strict one-core Linux run, prefix compute commands with
`taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
The native ONNX tools and Python trainer also use one inference/training thread.

Model-guided generation defaults to 16 concurrent games. Their pending leaf
positions share native ONNX batches of up to 32, avoiding a separate inference
call for every game. Use `--parallel-games 1` for a serial comparison. Seeds
are assigned to games before scheduling, records are written in game order,
and the provenance sidecar records concurrency and actual mean batch size.
Heuristic-only generation retains its original serial random stream.

Self-play writes one JSONL record per completed game. Each position contains a
280-float feature vector, an 81-way normalized root-visit policy target, and a
final win/draw/loss value from the player-to-move perspective. The trainer
splits whole games into train and validation partitions before applying board
rotations and reflections. It can expand all eight symmetries or sample one
random symmetry per position per epoch.

The default network is `280 → 256 ReLU → 128 ReLU`, followed by 81 policy logits
and one tanh value. Optional widths change only the two hidden layers; the ONNX
input/output interface and feature encoding stay fixed. Training masks illegal moves using the legal-move feature
channel. Inputs are own marks `[0:81]`, opponent marks `[81:162]`, legal moves
`[162:243]`, own local wins `[243:252]`, opponent local wins `[252:261]`, drawn
local boards `[261:270]`, and forced-board one-hot `[270:280]` (index 9 means no
forced board). ONNX outputs are raw `policy[batch,81]` logits and
`value[batch,1]` estimates for the player to move.

The exporter uses opset 17 and dynamic batch dimensions. Training writes the
model to `public/models/policy-value.onnx` and its run details, validation
losses, source dataset metadata, artifact size, and SHA-256 to
`public/models/metadata.json`. The dataset stays local and is ignored by git.

The bundled R16 checkpoint is a 280-input policy/value network with 512/256
hidden units and is 1,186,321 bytes. In 4,096 paired games it scored 55.03%
against R10 (97.5% interval 53.57–56.49%), establishing a head-to-head gain.
Against heuristic search, R16 scored 98.63% versus R10's 97.85%; the estimated
0.78-point difference remains uncertain (97.5% interval −0.30 to +1.86). See
`plans/training-2026-10-25-12h.md` for the full training summary and
`public/models/metadata.json` for the bundled checkpoint details.

Generation optimization improved a fixed 100-game, 24-simulation, seed-420
benchmark from 13.5 to 39.4 games/s on the development CPU, with byte-identical
datasets. On the learned model, native batching improved the same-size benchmark
from 12.96 to 38.65 games/s, also with byte-identical serial and concurrent
datasets. The concurrent run averaged 11.18 positions per inference call and
evaluated about 41,136 positions/s. These are single-process CPU measurements,
not GPU throughput estimates.
