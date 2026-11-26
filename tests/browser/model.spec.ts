import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

test('a corrupt model falls back to legal Hard computer play', async ({ page, context }) => {
  let corruptResponses = 0;
  await context.route('**/models/policy-value.onnx', async route => {
    await route.fulfill({ status: 200, contentType: 'application/octet-stream', body: 'invalid ONNX data' });
    corruptResponses++;
  });
  // Observe the actual worker reply before the UI can substitute a legal move.
  await page.addInitScript(() => {
    const replies: unknown[] = [];
    Object.assign(window, { hardMoveReplies: replies });
    window.Worker = class extends Worker {
      constructor(...args: ConstructorParameters<typeof Worker>) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (data && 'move' in data) replies.push(data);
        });
      }
    };
  });
  const response = await page.goto('./');
  expect(response?.ok()).toBe(true);
  const newGame = page.getByRole('button', { name: 'New game', exact: true });
  await newGame.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await dialog.getByRole('button', { name: 'Play vs computer' }).press('Enter');
  const difficulty = dialog.getByRole('slider', { name: 'Computer difficulty' });
  await difficulty.press('End');
  await expect(difficulty).toHaveAttribute('aria-valuetext', 'Hard');
  await dialog.getByRole('button', { name: 'X', exact: true }).press('Enter');
  await dialog.getByRole('button', { name: 'Start game' }).press('Enter');
  await page.locator('[data-board="4"][data-cell="2"]').press('Enter');
  await expect(page.locator('.computer-status')).toHaveAttribute('data-computer-source', 'search', { timeout: 25000 });
  expect(corruptResponses).toBeGreaterThan(0);
  const reply = await page.evaluate(() => (window as Window & {
    hardMoveReplies?: { move: number; source: string; simulations: number; settings: object }[]
  }).hardMoveReplies?.at(-1));
  expect(reply).toMatchObject({
    source: 'search', settings: { simulations: 384, timeMs: 220, tacticalProof: true },
  });
  expect(reply!.simulations).toBeGreaterThan(0);
  expect(reply!.simulations).toBeLessThanOrEqual(384);
  expect(Math.floor(reply!.move / 9)).toBe(2);
  await expect(page.locator(`[data-board="2"][data-cell="${reply!.move % 9}"]`)).toHaveClass(/cell-o/);
  await expect(page.locator('.cell-o')).toHaveCount(1);
  await expect(page.locator('.turn-title')).toHaveText('X to play');
});

test('cancelled workers cannot disturb a replacement computer request', async ({ page }) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const originalWorker = window.Worker;
    const workers: any[] = [];
    class FakeWorker {
      onmessage: ((event: { data: unknown }) => void) | null = null;
      onerror: (() => void) | null = null;
      request: any;
      constructor() { workers.push(this); }
      postMessage(request: unknown) { this.request = request; }
      terminate() {}
    }
    window.Worker = FakeWorker as unknown as typeof Worker;
    const clientPath = '/src/ai-client.ts';
    const gamePath = '/src/game.ts';
    const { ComputerPlayer } = await import(/* @vite-ignore */ clientPath);
    const { createGame } = await import(/* @vite-ignore */ gamePath);
    const computer = new ComputerPlayer();
    try {
      const first = computer.chooseMove(createGame(), 'easy').catch((error: Error) => error.name);
      computer.cancel();
      const second = computer.chooseMove(createGame(), 'easy');
      workers[0].onerror?.();
      workers[0].onmessage?.({ data: { id: workers[0].request.id, move: 0, source: 'search' } });
      workers[1].onmessage?.({ data: { id: workers[1].request.id, move: 40, source: 'model' } });
      return { first: await first, second: await second };
    } finally {
      computer.dispose();
      window.Worker = originalWorker;
    }
  });
  expect(result).toEqual({ first: 'AbortError', second: { move: 40, source: 'model' } });
});

test('browser ONNX outputs match offline Python inference across representative positions', async ({ page }) => {
  const fixture = JSON.parse(readFileSync(new URL('../../training/parity.json', import.meta.url), 'utf8'));
  await page.goto('/');
  const result = await page.evaluate(async (cases) => {
    const modulePath = '/src/onnx.ts';
    const { createEvaluator } = await import(/* @vite-ignore */ modulePath);
    const evaluate = await createEvaluator(new URL('models/policy-value.onnx', location.href).href);
    let maximumPolicyError = 0;
    let maximumValueError = 0;
    try {
      for (const item of cases) {
        const output = await evaluate(item.state);
        for (let action = 0; action < 81; action++) {
          maximumPolicyError = Math.max(maximumPolicyError, Math.abs(output.policy[action] - item.policy[action]));
        }
        maximumValueError = Math.max(maximumValueError, Math.abs(output.value - item.value));
      }
    } finally {
      await evaluate.dispose();
    }
    return { maximumPolicyError, maximumValueError };
  }, fixture.cases);
  expect(result.maximumPolicyError).toBeLessThan(0.0001);
  expect(result.maximumValueError).toBeLessThan(0.0001);
});
