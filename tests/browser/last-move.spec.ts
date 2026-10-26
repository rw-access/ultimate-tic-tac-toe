import { test, expect } from '@playwright/test';
import { createGame, legalBoards, play } from '../../src/game';

test('latest move follows both players, reload, undo, and a new game', async ({ page }) => {
  await page.goto('/');
  const repoLink = page.getByRole('link', { name: 'Source on GitHub' });
  await expect(repoLink).toBeVisible();
  await expect(repoLink).toHaveAttribute('href', 'https://github.com/rw-access/ultimate-tic-tac-toe');
  await expect(repoLink).toHaveAttribute('rel', 'noopener noreferrer');
  await expect(repoLink.locator('svg')).toHaveCount(1);
  const latest = page.locator('.cell-last-played');
  await expect(latest).toHaveCount(0);
  await page.locator('[data-board="4"][data-cell="4"]').click();
  await expect(latest).toHaveCount(1);
  await expect(latest).toHaveAttribute('aria-label', 'Board 5, square 5, X, last move');
  await page.locator('[data-board="4"][data-cell="0"]').click();
  await expect(latest).toHaveCount(1);
  await expect(latest).toHaveAttribute('aria-label', 'Board 5, square 1, O, last move');
  await page.reload();
  await expect(latest).toHaveAttribute('aria-label', 'Board 5, square 1, O, last move');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(latest).toHaveAttribute('aria-label', 'Board 5, square 5, X, last move');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(latest).toHaveCount(0);
  await page.locator('[data-board="4"][data-cell="4"]').click();
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  await page.getByRole('dialog', { name: 'New game' }).getByRole('button', { name: 'Start game' }).click();
  await expect(latest).toHaveCount(0);
});

test('computer moves have the same latest-move marker and undo restores it', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({ status: 404, body: 'use search fallback' }));
  await page.goto('/');
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  const setup = page.getByRole('dialog', { name: 'New game' });
  await setup.getByRole('button', { name: 'Play vs computer' }).click();
  await setup.getByRole('button', { name: 'O', exact: true }).click();
  await setup.getByRole('button', { name: 'Start game' }).click();
  const latest = page.locator('.cell-last-played');
  await expect(page.locator('.cell-mark')).toHaveCount(1, { timeout: 20000 });
  await expect(latest).toHaveCount(1);
  await expect(latest).toHaveAttribute('aria-label', /X, last move$/);
  const openingLabel = await latest.getAttribute('aria-label');
  await page.locator('.cell:enabled').first().click();
  await expect(page.locator('.cell-mark')).toHaveCount(3, { timeout: 20000 });
  await expect(latest).toHaveCount(1);
  await expect(latest).toHaveAttribute('aria-label', /X, last move$/);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.locator('.cell-mark')).toHaveCount(1);
  await expect(latest).toHaveAttribute('aria-label', openingLabel!);
  await expect(page.getByRole('link', { name: 'Source on GitHub' })).toBeVisible();
});

test('the latest move stays visible above a completed-board overlay', async ({ page }) => {
  let state = createGame();
  let seed = 12;
  while (!state.results.some(Boolean)) {
    const choices = legalBoards(state).flatMap(board => state.boards[board].flatMap((mark, cell) => mark === null ? [[board, cell]] : []));
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const [board, cell] = choices[seed % choices.length];
    const next = play(state, board, cell);
    if (next.results.some(Boolean)) {
      await page.goto('/');
      await page.evaluate(saved => localStorage.setItem('ultimate-tic-tac-toe:v1', JSON.stringify({ version: 1, state: saved, undo: [] })), state);
      await page.reload();
      await page.locator(`[data-board="${board}"][data-cell="${cell}"]`).click();
      const latest = page.locator('.cell-last-played');
      await expect(latest).toHaveCount(1);
      await expect(latest).toHaveAttribute('data-board', String(board));
      await expect(latest).toHaveAttribute('data-cell', String(cell));
      await expect(latest).toBeVisible();
      const marker = await latest.evaluate(element => ({ z: getComputedStyle(element).zIndex, dot: getComputedStyle(element, '::after').content, shadow: getComputedStyle(element).boxShadow }));
      expect(Number(marker.z)).toBeGreaterThan(0);
      expect(marker.dot).toBe('""');
      expect(marker.shadow).not.toBe('none');
      return;
    }
    state = next;
  }
  throw new Error('Expected a completed-board fixture');
});
