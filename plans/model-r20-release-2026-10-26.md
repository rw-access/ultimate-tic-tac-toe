# R20 model release — 2026-10-26

## Scope

Replace the bundled R16 ONNX weights with the latest passing neural checkpoint,
R20. Refresh native/browser parity fixtures, model metadata, and documentation.
The browser difficulty and endgame budgets remain as released in `b88c675`.

## Artifact and compatibility

- R20 SHA-256: `cd561330a1666ed3286f3f411277144c335f096b4105190864b5adc49054f135`.
- Previous R16 SHA-256: `d3b655d11dc6385ac4810658e640bcf0f074c1655993765bb54d8363d538a3d7`.
- Same 280-input, 512/256-hidden-unit policy/value network and 1,186,321-byte size.
- Input `state[batch,280]`; outputs `policy[batch,81]` and `value[batch,1]`.
- Existing feature encoding, rules, save format, and room protocol apply.

## Promotion evidence

Equal search effort against previously bundled R16: 64 simulations per side,
2,048 opening pairs with swapped seats, seed 2026102649. R20 recorded 1,441 wins,
1,377 draws, and 1,278 losses: **51.9897% score**, paired 95% interval
**50.8071%–53.1724%**. Score counts a draw as half a win.

The offline 1,024-simulation R20 search system scored **99.9589%** against the
full TypeScript policy/value heuristic at 64 simulations: 4,861 wins, two draws,
and one loss in 4,864 games. It also passed the preceding 512-simulation search
system at 62.1094% score in 1,024 games. These search-system results use different
budgets from the web runtime and do not establish browser difficulty win rates.

Compact benchmark evidence, configuration, seeds, hashes, and intervals are in
[r20-release-evaluation.json](artifacts/r20-release-evaluation.json). Training
provenance and export parity are in [metadata.json](../public/models/metadata.json).

## Validation and release

- Native parity: two tests passed across 12 representative positions.
- Browser parity, corrupt-model fallback, and cancellation: six tests passed on
  desktop and mobile. An initial run reused an existing R16 server; the corrected
  run used an isolated R20 server on port 4187 without disturbing it.
- Metadata, binary, and fixture SHA-256 match; the artifact remains under 1.5 MB.
- App/Worker TypeScript checks, Vite build, and Wrangler deployment dry run passed.
- Built production assets: desktop and mobile verified the R20 hash, legal
  model-backed replies at all three difficulties, and working hints; no page errors.
- Live model verification follows publication.

Publish directly to `main` and use the established Cloudflare Worker release.
If verification fails, restore the prior model, metadata, parity fixtures, and
model documentation together, then rebuild and redeploy through the same path.
