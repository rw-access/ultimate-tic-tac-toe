# Easy tactical floor and live analysis controls

## Goal

Keep Easy's policy-sampled style while letting it find proven forced wins a few
moves ahead and avoid immediate losses. Show the live Hint control when playing
the computer, keep active-game Analysis unavailable, and keep Game Review
available after the game.

## Design

- On Easy turns, use the existing exact endgame solver when it completes within
  its 8,000-node / nominal 12 ms budget.
- Otherwise, use a proof-only adversarial search through seven plies, ordered by
  the model policy at the root and bounded to 12,000 nodes / 24 ms. Return a
  move only when the line proves a win; an incomplete line returns no result.
- If neither search proves a result, sample the existing model policy from moves
  that do not allow an immediate opponent win. If no such move exists, retain
  the policy's full legal-move set.
- Show active-game Hint only when playing the computer; keep active-game
  Analysis unavailable. Keep Hint unavailable in two-player play. Preserve the
  post-game Review action and its analysis. Show a signed after-move position
  evaluation on Critical move cards, with one short key explaining that positive
  favors X and negative favors O.

## Acceptance criteria

- A proven win within seven plies takes priority over the sampled policy.
- Search exhaustion never counts as a proven win.
- Easy remains policy-sampled when no proof is found, with immediate-loss moves
  excluded when a safe move exists.
- Computer games can use Hint on the player's turn; two-player games do not
  expose Hint, and active gameplay does not expose Analysis. Completed games
  can still open and use Game Review.

## Limits

The capped proof search can miss forced wins when the tree does not fit its
budget. It never labels an incomplete line as solved.
