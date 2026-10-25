# Offline-trained browser opponent

## Scope

Add an optional computer opponent to the static game. Keep the TypeScript game
engine as the single source of rules for play, data generation, and evaluation.
Train locally in Python from the generated numeric examples;
ship a compact ONNX policy/value network and run ONNX Runtime Web on CPU in a
worker. Keep two-player mode, saves, undo, confirmation, and relative deployments.

## Contract

- Actions are `board * 9 + cell`, 0–80.
- Inputs are 280 float32 features from the current player's perspective:
  own marks 0–80; opponent marks 81–161; legal squares 162–242;
  own/opponent/drawn board results 243–269; forced destination 270–279
  (the final entry means unrestricted).
- Model: 280 → 256 ReLU → 128 ReLU; separate 81-logit policy and tanh value.
- ONNX input `state` `[batch, 280]`; outputs `policy` `[batch, 81]`,
  `value` `[batch, 1]`. Value is always relative to the player to move.
- Browser masks legality, searches with alternating value signs, and falls back
  to heuristic search on model failure. New game/undo/mode changes cancel work.
- Generate MCTS targets and game results, split by game before symmetry
  augmentation, train, export, and evaluate against fixed baselines in both seats.
- Offline self-play/evaluation runs native `onnxruntime-node` in the same process
  as TypeScript search. Python/PyTorch only reads datasets and exports model
  files between rounds; there is no per-position subprocess or JSON bridge.

## Acceptance

- Rules and encoding agree across Python and TypeScript fixtures.
- Search obeys legality, terminal positions, and immediate game wins.
- Python and browser ONNX outputs agree within float32 tolerance.
- Real browser model loading, move replies, fallback, undo, cancellation, and
  preference restoration work on desktop and mobile.
- Unit tests, Python tests, production build, browser tests, and a production
  deployment beneath a URL prefix pass.
- Bundle the trained checkpoint with its provenance and measured metrics;
  report pilot limitations without assuming it beats its search teacher.

## Progress

The initial 1,000-game dataset was generated at 24 simulations per position in
72.42 seconds on two available CPU cores using one process. Search optimization
improved a fixed 100-game benchmark from 13.5 to 39.4 games/s while preserving
byte-identical data. Native ONNX batching improved learned-model self-play from
12.96 to 38.65 games/s at the same budget, with identical serial/concurrent
datasets and a mean batch of 11.18 positions. The 462,865-byte checkpoint has
been exported and its Torch and ONNX outputs agree within 1.2e-6. Unit, Python,
desktop/mobile browser, and production URL-prefix checks pass. The desktop
layout also keeps its controls and board within a 720px-high viewport.

The pilot has not surpassed heuristic search: raw greedy policy scored 0.62
against random play, and model-guided search scored 0.22 against heuristic
search at 32 simulations, in 50 balanced-seat games per matchup. Treat the
checkpoint as experimental; the pipeline supports further self-play rounds.
Subsequent user requests authorized iterative one-core training, integrating
the best checkpoint, and publishing the static app through the existing hosting
path. See `training-2026-10-24.md` for the final model and evaluation results;
hosting records stay outside this repository.
