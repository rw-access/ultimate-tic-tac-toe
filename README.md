# Ultimate Tic-Tac-Toe

A browser game for local two-player play, invite-only online rooms, or one person
against a computer. The game and AI are written in TypeScript and run in the
browser; online rooms use a small Cloudflare Worker backend. No accounts or
matchmaking are required.

Win three little boards in a row to win the big board. Your square sends your opponent to the matching little board. If that board has already finished, they may choose any open board. Drawn boards close without belonging to either player.

Tap a square once to play. The **+** button opens the new-game menu for local two-player, online, or computer play. Online games create an eight-character room code and a shareable link; one opponent joins, and turns are checked by the room server. In computer games, choose difficulty and play as X, O, or a random side. The **?** button opens the rules page. **Hint** highlights one suggested move. **Analysis** opens the estimated X/O advantage and best line, with an alternative line tucked away. The estimate is not calibrated win odds. After a game, **Review game** checks your moves for missed immediate wins and replies that allow an immediate loss. Undo and browser-local saves make it easy to pick up where you left off.

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

`npm run dev` serves the static game at the Vite root. Use `npm run dev:worker`
to build it and run the Worker, room API, and Durable Objects locally at
`/play/ultimate-tic-tac-toe/`. The Worker serves the static assets and API from that
path on one origin.

## Checks

```sh
npm test
npm run test:worker
npm run build
npx playwright install chromium
npm run test:browser
npm run test:production
```

`npm run build` produces static files under `dist/play/ultimate-tic-tac-toe/`, with
relative asset URLs. `npm run preview` previews the build at
`/play/ultimate-tic-tac-toe/`.

`wrangler.jsonc` configures Cloudflare Workers Static Assets and one SQLite-backed
Durable Object per room. Rooms are authoritative, accept only legal moves from
the correct seat, reconnect using a browser-stored seat token, and expire after
24 hours without activity. Only the room code is in the invite URL; seat tokens
are not shared with the opponent.

## Cloudflare deployment

The deploy script builds and type-checks both the browser app and Worker before
calling Wrangler. It never runs as part of tests or the ordinary build.

```sh
npm run deploy:worker -- --dry-run
npm run deploy:worker
```

The Worker name can also come from `UTTT_WORKER_NAME`. The production route is
`rw-access.dev/play/ultimate-tic-tac-toe/`, configured in `wrangler.jsonc`.
The portfolio at `rw-access.dev/play/` provides the game catalog. The Worker
serves only the canonical `/play/ultimate-tic-tac-toe/` prefix, and the build
publishes only that asset tree.
Authenticate Wrangler with `wrangler login`, or provide `CLOUDFLARE_API_TOKEN`
and `CLOUDFLARE_ACCOUNT_ID` in the environment.

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

The bundled checkpoint is R16: a policy/value network with 280 inputs and
512/256 hidden units, exported as a 1,186,321-byte ONNX file (about 1.13 MiB).
Inputs include mini-board wins and draws, legal cells, and the forced
destination. Hard mode searches in a web worker with a 140 ms search budget;
model loading and the first inference take longer.

R16 scored 55.03% against R10 over 4,096 paired games (97.5% interval
53.57%–56.49%). Against heuristic search it scored 98.63%, compared with
R10's 97.85%; the estimated 0.78-point gain remains uncertain (97.5% interval
−0.30 to +1.86). These results support the direct R16-over-R10 choice, not a
proven heuristic-strength increase. See the [session report](plans/training-2026-10-25-12h.md)
for details and limits.

Run the Python checks after installing the training dependencies:

```sh
.venv/bin/python -m unittest discover -s training -p 'test_*.py'
```
