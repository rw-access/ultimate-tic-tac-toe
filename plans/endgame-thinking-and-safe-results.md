# Endgame thinking and safe results

Requested 2026-10-26: give late positions more time, retain proved wins while
looking for shorter wins, and prefer guaranteed draws to uncertain alternatives.

Computer turns with at most 18 empty cells on open local boards try exact search
before loading the model. The cap is 32,000 nodes; time allowances are 500 ms on
Easy, 1,000 ms on Medium and 2,000 ms on Hard. Ordinary early-game difficulty
settings and Easy's separate seven-ply tactical scan remain unchanged. Hints use
250 ms/32k; analysis 500 ms/32k; batch review keeps 12 ms/8k per position.
The existing Web Worker performs this work so the board remains responsive.

Mate-distance negamax prefers shorter wins and delays unavoidable losses. A
completed winning move survives budget exhaustion even if its optimal distance
is not yet proved. A completed drawing move can also survive as a guaranteed
floor; the root remains unsolved because an unexplored move might win. Opponent
mistakes can still allow a win after choosing a move that guarantees a draw.
Unproved partial bounds never become a selectable guarantee. Analysis reports
these partial floors as search estimates, not exact root draws.

This changes runtime search, not model weights. The separate R20 search
experiments use frozen source/configurations and do not measure this runtime
profile or its added draw-retention behavior.

Validation: eight focused endgame tests and the application/worker build passed.
An independent oracle enumerated 121,171 states across 101 saved late positions
(80 draws, 13 wins, 8 losses). All 101 full solves and 1,010 capped searches
agreed on proved outcomes and safety. The audit checked 620 guarantees,
including 139 interrupted draw fallbacks and three interrupted win fallbacks,
with zero unsafe proofs. Raw audit artifacts remain under
`/home/rw/metatictactoe-diagnostics-20261026/`.
