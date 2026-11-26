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
The model weights are 1,476,740 bytes (1.48 MB); the ONNX WebAssembly runtime adds about
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

The bundled Gen102 model scored **57.2% against Gen101** across 512 games on the primary client (140 wins, 306 draws, 66 losses, paired 95% interval 54.4%–60.0%). On the second client it scored 62.7% across 256 games (94 wins, 133 draws, 29 losses, paired 95% interval 58.7%–66.5%). Both used the production Hard worker with up to 384 simulations and 220 ms of tree search.

Against the original uttt.ai Quick engine at 1,000 visits, Gen102 scored **38.0%** (8 wins, 60 draws, 32 losses) and Gen101 scored 35.0% (4 wins, 62 draws, 34 losses). Each played both seats on all 50 published openings. A draw counts as half a point. [Match report](plans/model-strength-2026-11-26-morning.md)

Gen102 received 1,200 training updates starting from Gen101. Training used searched positions, proved outcomes and moves where the student and its teacher disagreed. The decoder computes the policy only for legal moves. Five game phases have separately fitted policy and value scales. [Model metadata](public/models/metadata.json) · [Upstream model and data](https://github.com/arnowaczynski/utttai) · [Apache-2.0 license](public/models/utttai-Apache-2.0.txt)

The afternoon checkpoint compared each candidate with the model that shipped that morning. In fresh matches against their predecessors, the 512 KB model scored 56.5% over 384 games and the 5 MB model scored 53.8% over 384 games. [Afternoon results and training report](plans/model-strength-2026-11-26-afternoon.md)

TypeScript generates games and plays both seats using the browser rules and native ONNX inference. Python trains on the resulting positions and exports the models. Offline weights for the other six size classes are separate downloads. [Training commands](training/README.md) · [Model downloads](https://github.com/rw-access/ultimate-tic-tac-toe/releases/latest)

Run the Python checks after installing the training dependencies:

```sh
.venv/bin/python -m unittest discover -s training -p 'test_*.py'
```
