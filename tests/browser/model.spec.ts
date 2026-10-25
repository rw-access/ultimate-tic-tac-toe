import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

test('a corrupt model falls back to legal computer play', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({
    status: 200, contentType: 'application/octet-stream', body: 'invalid ONNX data',
  }));
  await page.goto('/');
  await page.getByRole('button', { name: 'New game' }).click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await dialog.getByRole('button', { name: 'Play vs computer' }).click();
  await dialog.getByRole('button', { name: 'Start game' }).click();
  await page.locator('[data-board="4"][data-cell="2"]').click();
  await expect(page.locator('[data-computer-source="search"]')).toBeVisible({ timeout: 25000 });
  await expect(page.locator('[data-mini-board="2"] .cell-o')).toHaveCount(1);
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
