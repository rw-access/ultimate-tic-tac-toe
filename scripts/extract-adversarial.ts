#!/usr/bin/env node
/** Extracts high-quality training positions from external-arena tournament JSONL files. */
import { readFileSync, writeFileSync } from 'node:fs';
import { createGame, play, type Player } from '../src/game.ts';
import { encodeState, legalMoves } from '../src/ai.ts';

function main() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.error('Usage: extract-adversarial.ts <output.jsonl> <input1.jsonl> [input2.jsonl ...]');
    process.exit(1);
  }
  const [outputPath, ...inputPaths] = args;
  let totalGames = 0;
  let totalPositions = 0;
  const outLines: string[] = [];
  const seenGames = new Set<string>();

  for (const inputPath of inputPaths) {
    let content = '';
    try {
      content = readFileSync(inputPath, 'utf8');
    } catch (err) {
      console.warn(`Could not read ${inputPath}: ${err}`);
      continue;
    }
    const lines = content.trim().split('\n').filter(Boolean);
    // skip config header if present
    for (let i = 0; i < lines.length; i++) {
      let row: any;
      try {
        row = JSON.parse(lines[i]);
      } catch {
        continue;
      }
      if (row.format && row.format.startsWith('external-arena')) continue;

      for (const seatKey of ['modelX', 'modelO'] as const) {
        const match = row[seatKey];
        if (!match || !Array.isArray(match.moves) || match.moves.length === 0) continue;
        const winner: Player | 'draw' | null = match.winner;
        if (!winner) continue;

        const moves: number[] = match.moves;
        const moveKey = moves.join(',');
        if (seenGames.has(moveKey)) continue;
        seenGames.add(moveKey);

        let state = createGame();
        const samples: Array<[number[], number[], number]> = [];

        for (let mIdx = 0; mIdx < moves.length; mIdx++) {
          const move = moves[mIdx];
          const turn = state.turn;
          const legal = legalMoves(state);
          if (!legal.includes(move)) break;

          // Only learn policy from winning moves or drawn moves (never losing moves)
          const isWinner = winner === turn;
          const isDraw = winner === 'draw';

          if (isWinner || isDraw) {
            const pTarget = new Array<number>(81).fill(0);
            pTarget[move] = 1.0;
            const outcome = isWinner ? 1.0 : 0.0;
            const encoded = Array.from(encodeState(state));
            samples.push([encoded, pTarget, outcome]);
          }

          state = play(state, Math.floor(move / 9), move % 9);
        }

        if (samples.length > 0) {
          outLines.push(JSON.stringify({ game: totalGames++, winner, moves, samples }));
          totalPositions += samples.length;
        }
      }
    }
  }

  writeFileSync(outputPath, outLines.join('\n') + '\n');
  console.log(`Extracted ${totalGames} unique adversarial games (${totalPositions.toLocaleString()} positions) -> ${outputPath}`);
}

main();
