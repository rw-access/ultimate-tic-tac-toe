# Ultimate Tic-Tac-Toe

A browser game for two people sharing a screen, or one person against a computer.
Built with TypeScript and Vite; no backend or accounts needed.

Win three little boards in a row to win the big board. Your square sends your opponent to the matching little board. If that board has already finished, they may choose any open board. Drawn boards close without belonging to either player.

Tap a square once to play. The **+** button opens the new-game menu for two players or computer play; in computer games, choose difficulty and play as X, O, or a random side. The **?** button opens the rules page. **Hint** highlights one suggested move. **Analysis** opens the estimated X/O advantage and best line, with an alternative line tucked away. The estimate is not calibrated win odds. After a game, **Review game** checks your moves for missed immediate wins and replies that allow an immediate loss. Undo and browser-local saves make it easy to pick up where you left off.

Choose **Computer** in the new-game menu to play against the model. Easy samples
the trained model's move preferences; Medium and Hard add progressively larger
Monte Carlo tree searches. Starting a new game applies the selected mode and
resets the board. Undo takes back your move and the computer's reply.
Medium and Hard also check immediate global wins and avoid allowing an
immediate global loss whenever a safe move exists.

The computer and optional analysis run ONNX Runtime Web on CPU in Web Workers.
Model and runtime assets load only when the computer needs to move or you ask
for a hint, analysis, or review, and are served by the same static site; no AI service, GPU, or
cross-origin isolation is required. If the model cannot load or run, the
computer uses basic search and shows that status.
The model weights are about 1.2 MB; the ONNX WebAssembly runtime adds about
14.2 MB uncompressed (about 3.7 MB with gzip). Enable HTTP compression when
hosting. This does not include a service worker for offline page reloads.

## Development

Requires Node.js 22.18+ or 24+.

```sh
npm ci
npm run dev
```

## Checks

```sh
npm test
npm run build
npx playwright install chromium
npm run test:browser
npm run test:production
```

`npm run build` produces portable static files in `dist/`, with relative asset URLs so the game works at a site root or under a URL prefix. Serve the directory with any static web server. `npm run preview` previews a build locally.

Games and preferences stay in your browser's local storage. Players on separate devices have independent games; this is a pass-and-play game, without network multiplayer.

## Offline training

The bundled model is trained locally from generated games. TypeScript generates
games and evaluates both playing seats using the same rules and search engine
as the browser, with native ONNX Runtime for local model inference. Python
consumes numeric examples, trains with board symmetries, and exports ONNX.
Model-guided self-play can provide another round of examples. The runtimes
exchange dataset/model files between rounds; search never calls a Python
subprocess for individual positions.
See [training/README.md](training/README.md) for setup and commands, and
[public/models/metadata.json](public/models/metadata.json) for the bundled
checkpoint's provenance and measured results. Training data and Python
environments stay out of Git.

The bundled checkpoint is R10: a policy/value network with 280 inputs and
512/256 hidden units, exported as a 1,186,321-byte ONNX file (about 1.13 MiB).
Inputs include mini-board wins and draws, legal cells, and the forced
destination. Hard mode searches in a web worker with a 140 ms search budget;
model loading and the first inference take longer.

At 64 simulations per move, a fixed independent 4,096-game match against R09
scored 54.88% (paired 95% interval 53.61%–56.15%; draws count half). R10 also
scored 97.80% against heuristic search over 1,024 games at equal 64-simulation
budgets; R09 scored 95.26% on the same paired cases. A separate 64-game browser
match at the actual Hard setting initially scored 55.47%. A larger fresh
128-pair browser match (256 games, seed 19106142) scored 57.81% (89 wins, 118
draws, 49 losses; paired 95% interval 53.54%–62.08%). It used the unchanged
R10 Hard worker at 140 ms and is reported separately from the earlier small
sample. Warm responses had a 140.5 ms median and 141 ms p95 on the test
machine; physical devices can differ. This supports an R10 edge in this browser
setting, not a general playing-strength rating.
To check the apparent X advantage, an R10 mirror match started 512 games from
the empty board at 64 simulations per move; all 512 ended in draws, with no X
or O wins observed. This gives no evidence of a first-player edge in that
sample, but does not prove perfect-play symmetry.

The current fit used a rolling replay window of about 18,400 games / 900,000
positions. The longer session recorded 86,550 unique self-play games, including
900 named holdout games and experimental branches. See
[the session plan](plans/training-2026-10-25-12h.md) for results and limitations.

Run the Python checks after installing the training dependencies:

```sh
.venv/bin/python -m unittest discover -s training -p 'test_*.py'
```
