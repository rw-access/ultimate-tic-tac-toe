import { test, expect } from '@playwright/test';
import { createGame, play, legalBoards } from '../../src/game';

test('single tap routes play, focuses next board, persists and undoes', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.cell:enabled')).toHaveCount(81);
  await page.locator('[data-board="4"][data-cell="2"]').click();
  await expect(page.locator('.turn-title')).toHaveText('O to play');
  await expect(page.locator('.cell:enabled')).toHaveCount(9);
  await expect(page.locator('[data-mini-board="2"]')).toHaveClass(/is-forced/);
  await expect(page.locator('[data-mini-board="2"] .cell').first()).toBeFocused();
  await page.reload();
  await expect(page.locator('.turn-title')).toHaveText('O to play');
  await expect(page.locator('[data-board="4"][data-cell="2"]')).toContainText('X');
  await page.getByRole('button', { name: 'Undo' }).click();
  await expect(page.locator('.cell:enabled')).toHaveCount(81);
});

test('moves are single-tap and the game and rules pages are easy to open', async ({ page }) => {
  await page.goto('/');
  const cell = page.locator('[data-board="4"][data-cell="4"]');
  await cell.click();
  await expect(page.locator('.turn-title')).toHaveText('O to play');
  await expect(page.getByRole('checkbox')).toHaveCount(0);
  await page.getByRole('button', { name: /New game/ }).first().click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Cancel' }).click();
  await expect(page.locator('.turn-title')).toHaveText('O to play');
  await page.getByRole('button', { name: /New game/ }).first().click();
  await page.getByRole('dialog', { name: 'New game' }).getByRole('button', { name: 'Start game' }).click();
  await expect(page.locator('.cell:enabled')).toHaveCount(81);

  await page.getByRole('button', { name: 'How to play' }).click();
  const rules = page.getByRole('dialog', { name: 'How to play' });
  await expect(rules.locator('.rules-list li')).toHaveCount(5);
  await expect(rules.getByRole('link', { name: /Wikipedia/ })).toHaveAttribute('href', 'https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe');
  await rules.getByRole('button', { name: 'Got it' }).click();
});

test('legal sequence creates a win overlay and keeps finished boards closed', async ({ page }) => {
  let s = createGame();
  let seed = 12;
  while (!s.results.some(Boolean)) {
    const choices = legalBoards(s).flatMap(b => s.boards[b].flatMap((c,i) => c === null ? [[b,i]] : []));
    seed = (Math.imul(seed,1664525) + 1013904223) >>> 0;
    const [b,c] = choices[seed % choices.length];
    const next = play(s,b,c);
    if (next.results.some(Boolean)) {
      await page.goto('/');
      await page.evaluate(state => localStorage.setItem('ultimate-tic-tac-toe:v1', JSON.stringify({ version:1, state, undo: [] })), s);
      await page.reload();
      await page.locator(`[data-board="${b}"][data-cell="${c}"]`).click();
      const board = page.locator(`[data-mini-board="${b}"]`);
      await expect(board.locator('.result-overlay')).toBeVisible();
      await expect(board.locator('.cell:enabled')).toHaveCount(0);
      await expect(board).toHaveClass(/is-complete/);
      await expect(board).toHaveClass(/just-completed/);
      await page.locator('.cell:enabled').first().click();
      await expect(board).not.toHaveClass(/just-completed/);
      await page.reload();
      await expect(board.locator('.result-overlay')).toBeVisible();
      await expect(board).not.toHaveClass(/just-completed/);
      return;
    }
    s = next;
  }
});

test('layout stays usable and malformed storage remains playable', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem('ultimate-tic-tac-toe:v1', '{invalid'));
  await page.reload();
  await expect(page.locator('.cell:enabled')).toHaveCount(81);
  const layout = await page.evaluate(() => ({
    board: document.querySelector('.board-frame')!.getBoundingClientRect().width,
    width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
    height: innerHeight,
  }));
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width);
  if (layout.width <= 600) expect(layout.board).toBeGreaterThanOrEqual(layout.width - 32);
  else {
    expect(layout.scrollHeight).toBeLessThanOrEqual(layout.height);
    if (layout.width >= 900) expect(layout.board).toBeGreaterThanOrEqual(440);
  }
  await page.getByRole('button', { name: 'How to play', exact: true }).first().click();
  await expect(page.locator('#how-to-dialog')).toBeVisible();
  await page.getByRole('button', { name: 'Got it' }).click();
  expect(errors).toEqual([]);
});

test('phone board stays wide and setup controls stay inside the new-game menu', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 664 });
  await page.goto('/');
  await page.getByRole('button', { name: 'New game' }).click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await dialog.getByRole('button', { name: 'Play vs computer' }).click();
  const layout = await page.evaluate(() => ({
    board: document.querySelector('.board-frame')!.getBoundingClientRect().width,
    width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
    height: innerHeight,
  }));
  expect(layout.board).toBeGreaterThanOrEqual(layout.width - 32);
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width);

  const slider = page.getByRole('slider', { name: 'Computer difficulty' });
  await expect(slider).toBeVisible();
  await dialog.getByRole('button', { name: 'Start game' }).click();
  await expect(page.getByRole('slider', { name: 'Computer difficulty' })).toHaveCount(0);
  await page.locator('.cell:enabled').first().click();
  await expect(page.locator('.cell-x')).toHaveCount(1);
});

test('phone controls use readable type and center the Undo icon', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  const metrics = await page.evaluate(() => {
    const fontSize = (selector: string) => Number.parseFloat(getComputedStyle(document.querySelector(selector)!).fontSize);
    const undo = document.querySelector<HTMLButtonElement>('[data-action="undo"]')!;
    const icon = undo.querySelector('svg')!;
    const pathBox = icon.querySelector('path')!.getBBox();
    const viewBox = icon.viewBox.baseVal;
    const buttonBox = undo.getBoundingClientRect();
    const iconBox = icon.getBoundingClientRect();
    return {
      status: fontSize('.computer-status'),
      hint: fontSize('[data-action="hint"]'),
      scoreLabel: fontSize('.score span'),
      undoCenterOffset: {
        x: Math.abs((buttonBox.left + buttonBox.right - iconBox.left - iconBox.right) / 2),
        y: Math.abs((buttonBox.top + buttonBox.bottom - iconBox.top - iconBox.bottom) / 2),
      },
      undoGlyphOffset: {
        x: Math.abs(pathBox.x + pathBox.width / 2 - viewBox.width / 2),
        y: Math.abs(pathBox.y + pathBox.height / 2 - viewBox.height / 2),
      },
    };
  });
  expect(metrics.status).toBeGreaterThanOrEqual(14);
  expect(metrics.hint).toBeGreaterThanOrEqual(13);
  expect(metrics.scoreLabel).toBeGreaterThanOrEqual(15);
  expect(metrics.undoCenterOffset.x).toBeLessThanOrEqual(0.5);
  expect(metrics.undoCenterOffset.y).toBeLessThanOrEqual(0.5);
  expect(metrics.undoGlyphOffset.x).toBeLessThanOrEqual(0.5);
  expect(metrics.undoGlyphOffset.y).toBeLessThanOrEqual(0.5);
});
