import assert from 'node:assert/strict';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import type { Browser } from '@playwright/test';
import { makeAgent } from '../scripts/browser-arena.ts';
import { createGame, play, type GameState } from '../src/game.ts';

test('only the first empty-board Easy probe receives the cold-start allowance', async () => {
  const posted: Record<string, unknown>[] = [];
  const delays: number[] = [];
  const pageWindow = {
    setTimeout(callback: () => void, ms: number) { delays.push(ms); queueMicrotask(callback); return 1; },
    clearTimeout() {},
  } as Window & { __browserArena: { request(state: GameState, difficulty: string): Promise<unknown> } };
  class Worker {
    addEventListener() {}
    postMessage(message: Record<string, unknown>) { posted.push(message); }
  }
  const page = {
    on() {}, async goto() {},
    evaluate(fn: Function, args: unknown) {
      return runInNewContext(`(${fn.toString()})`, { Worker, window: pageWindow, performance })(args);
    },
  };
  const browser = {
    async newContext() { return { async newPage() { return page; }, async route() {} }; },
  } as unknown as Browser;
  const start = () => makeAgent('challenger', '', '', Buffer.alloc(0), 'http://example.test/', 'worker.js', browser);
  const empty = createGame();
  const opening = play(empty, 4, 4);
  await start();
  await assert.rejects(pageWindow.__browserArena.request(empty, 'easy'), /90000 ms on cold start request 1/);
  await assert.rejects(pageWindow.__browserArena.request(empty, 'easy'), /20000 ms on easy request 2/);
  await assert.rejects(pageWindow.__browserArena.request(opening, 'hard'), /20000 ms on hard request 3/);
  await start();
  await assert.rejects(pageWindow.__browserArena.request(empty, 'hard'), /20000 ms on hard request 1/);
  await start();
  await assert.rejects(pageWindow.__browserArena.request(opening, 'easy'), /20000 ms on easy request 1/);
  assert.deepEqual(delays, [90_000, 20_000, 20_000, 20_000, 20_000]);
  assert.deepEqual(Object.keys(posted[0]).sort(), ['action', 'difficulty', 'id', 'modelUrl', 'state']);
  assert.deepEqual(posted.map(message => message.difficulty), ['easy', 'easy', 'hard', 'hard', 'easy']);
});
