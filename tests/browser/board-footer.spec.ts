import { test, expect } from '@playwright/test';
import { createGame, legalBoards, play, type GameState } from '../../src/game';

async function checkFooter(page: import('@playwright/test').Page) {
  const footer = await page.evaluate(() => {
    const board = document.querySelector('.board-frame')!.getBoundingClientRect();
    const map = document.querySelector('.route-map')!.getBoundingClientRect();
    const buttons = [...document.querySelectorAll<HTMLElement>('.under-board button')].map(button => {
      const bounds = button.getBoundingClientRect();
      return { fontSize: parseFloat(getComputedStyle(button).fontSize), left: bounds.left, right: bounds.right, height: bounds.height };
    });
    const repo = document.querySelector('.repo-link')!.getBoundingClientRect();
    return { width: innerWidth, scrollWidth: document.documentElement.scrollWidth, centerDelta: Math.abs((board.left + board.right) / 2 - (map.left + map.right) / 2), buttons, repoRight: repo.right, mapLeft: map.left };
  });
  expect(footer.centerDelta).toBeLessThanOrEqual(1);
  expect(footer.scrollWidth).toBeLessThanOrEqual(footer.width + 1);
  expect(footer.repoRight).toBeLessThan(footer.mapLeft);
  for (const button of footer.buttons) {
    expect(button.fontSize).toBeGreaterThanOrEqual(14);
    expect(button.height).toBeGreaterThanOrEqual(36);
    expect(button.left).toBeGreaterThanOrEqual(0);
    expect(button.right).toBeLessThanOrEqual(footer.width);
  }
}

test('footer stays centered with asymmetric computer controls and larger text', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({ status: 404, body: 'use search fallback' }));
  await page.goto('/');
  for (const width of [320, 375, 390, 1280, 1440]) {
    await page.setViewportSize({ width, height: width < 600 ? 844 : 900 });
    await checkFooter(page);
  }
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  const setup = page.getByRole('dialog', { name: 'New game' });
  await setup.getByRole('button', { name: 'Play vs computer' }).click();
  await setup.getByRole('button', { name: 'X', exact: true }).click();
  await setup.getByRole('button', { name: 'Start game' }).click();
  await page.getByRole('button', { name: 'Hint', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Hint shown', exact: true })).toBeVisible({ timeout: 20000 });
  for (const width of [320, 375, 390, 1280, 1440]) {
    await page.setViewportSize({ width, height: width < 600 ? 844 : 900 });
    await checkFooter(page);
    const gap = await page.evaluate(() => {
      const map = document.querySelector('.route-map')!.getBoundingClientRect();
      const tools = document.querySelector('.board-tools')!.getBoundingClientRect();
      return tools.left - map.right;
    });
    expect(gap).toBeGreaterThanOrEqual(0);
  }
});

test('finished games and game review keep the footer centered', async ({ page }) => {
  let state = createGame();
  const undo: GameState[] = [];
  let seed = 8917;
  while (!state.winner) {
    const moves = legalBoards(state).flatMap(board => state.boards[board].flatMap((mark, cell) => mark === null ? [board * 9 + cell] : []));
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    undo.push(state);
    const move = moves[seed % moves.length];
    state = play(state, Math.floor(move / 9), move % 9);
  }
  await page.goto('/');
  await page.evaluate(saved => localStorage.setItem('ultimate-tic-tac-toe:v1', JSON.stringify({ version: 1, ...saved })), { state, undo });
  await page.reload();
  for (const width of [320, 390, 1440]) {
    await page.setViewportSize({ width, height: width < 600 ? 844 : 900 });
    await checkFooter(page);
  }
  await page.getByRole('button', { name: 'Review game', exact: true }).click();
  for (const width of [320, 390, 1440]) {
    await page.setViewportSize({ width, height: width < 600 ? 844 : 900 });
    await checkFooter(page);
  }
});
