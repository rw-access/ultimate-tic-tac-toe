import { test, expect } from '@playwright/test';
import { createGame, play } from '../../src/game';

const THEME_KEY = 'ultimate-tic-tac-toe:theme';

test('Auto follows the system, including changes while the game is open', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Theme: Auto. Switch to Light', exact: true })).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.getByRole('button', { name: 'Theme: Auto. Switch to Light', exact: true })).toBeVisible();
  expect(await page.evaluate(key => localStorage.getItem(key), THEME_KEY)).toBeNull();
});

test('theme control cycles, remembers explicit choices, and resets to Auto without changing play', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  await page.locator('[data-board="4"][data-cell="2"]').click();
  const boardBefore = await page.locator('.board-frame').boundingBox();
  const theme = page.locator('[data-action="theme"]');
  await theme.click();
  await expect(theme).toHaveText('Light');
  await expect(theme).toBeFocused();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.reload();
  await expect(theme).toHaveText('Light');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.emulateMedia({ colorScheme: 'light' });
  await theme.focus();
  await page.keyboard.press('Enter');
  await expect(theme).toHaveText('Dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(theme).toHaveText('Dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await theme.click();
  await expect(theme).toHaveText('Auto');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(await page.evaluate(key => localStorage.getItem(key), THEME_KEY)).toBeNull();
  await expect(page.locator('.turn-title')).toHaveText('O to play');
  await expect(page.locator('[data-board="4"][data-cell="2"]')).toContainText('X');
  await expect(page.locator('.cell:enabled')).toHaveCount(9);
  expect(await page.locator('.board-frame').boundingBox()).toEqual(boardBefore);
});

test('both palettes keep marks distinct and theme controls within the existing layout', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-board="4"][data-cell="2"]').click();
  await page.locator('[data-board="2"][data-cell="0"]').click();
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expect(page.locator('html')).toHaveAttribute('data-theme', colorScheme);
    const metrics = await page.evaluate(() => {
      const color = (selector: string) => getComputedStyle(document.querySelector(selector)!).color;
      const luminance = (rgb: string) => {
        const values = rgb.match(/\d+/g)!.slice(0, 3).map(value => {
          const channel = Number(value) / 255;
          return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722;
      };
      const contrast = (selector: string) => {
        const mark = document.querySelector(selector)!;
        const foreground = luminance(getComputedStyle(mark).color);
        const background = luminance(getComputedStyle(mark.parentElement!).backgroundColor);
        return (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05);
      };
      return {
        x: color('.mark-x'), o: color('.mark-o'),
        xContrast: contrast('.mark-x'), oContrast: contrast('.mark-o'),
        background: getComputedStyle(document.documentElement).backgroundColor,
        scrollWidth: document.documentElement.scrollWidth, width: innerWidth,
        board: document.querySelector('.board-frame')!.getBoundingClientRect().width,
        theme: document.querySelector('[data-action="theme"]')!.getBoundingClientRect().toJSON(),
      };
    });
    expect(metrics.x).not.toBe(metrics.o);
    expect(metrics.xContrast).toBeGreaterThanOrEqual(4.5);
    expect(metrics.oContrast).toBeGreaterThanOrEqual(4.5);
    expect(metrics.x).not.toBe(metrics.background);
    expect(metrics.o).not.toBe(metrics.background);
    expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.width);
    expect(metrics.theme.x).toBeGreaterThanOrEqual(0);
    expect(metrics.theme.right).toBeLessThanOrEqual(metrics.width);
    if (metrics.width <= 600) expect(metrics.board).toBeGreaterThanOrEqual(metrics.width - 32);
  }
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'New game', exact: true })).toBeVisible();
});

test('invalid theme storage falls back to Auto', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  await page.evaluate(key => localStorage.setItem(key, 'invalid'), THEME_KEY);
  await page.reload();
  await expect(page.locator('[data-action="theme"]')).toHaveText('Auto');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});

test('rebrand migration preserves saved games and preferences', async ({ page }) => {
  const state = play(createGame(), 4, 2);
  const save = JSON.stringify({ version: 1, state, humanSide: 'X', undo: [createGame()], computerMoves: [] });
  const opponent = JSON.stringify({ mode: 'two-players', difficulty: 'hard', sideChoice: 'O' });
  await page.goto('/');
  await page.evaluate(({ game, preference }) => {
    localStorage.setItem('meta-tic-tac-toe:v1', game);
    localStorage.setItem('meta-tic-tac-toe:opponent', preference);
    localStorage.setItem('meta-tic-tac-toe:theme', 'dark');
  }, { game: save, preference: opponent });
  await page.reload();

  await expect(page.locator('.turn-title')).toHaveText('O to play');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  expect(await page.evaluate(() => localStorage.getItem('ultimate-tic-tac-toe:v1'))).toBe(save);
  expect(await page.evaluate(() => localStorage.getItem('ultimate-tic-tac-toe:opponent'))).toBe(opponent);
  expect(await page.evaluate(() => localStorage.getItem('ultimate-tic-tac-toe:theme'))).toBe('dark');
  expect(await page.evaluate(() => ['meta-tic-tac-toe:v1', 'meta-tic-tac-toe:opponent', 'meta-tic-tac-toe:theme'].map(key => localStorage.getItem(key)))).toEqual([null, null, null]);

  await page.getByRole('button', { name: 'New game' }).click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await expect(dialog.getByRole('button', { name: 'Two players' })).toHaveAttribute('aria-pressed', 'true');
  await dialog.getByRole('button', { name: 'Play vs computer' }).click();
  await expect(dialog.getByRole('slider', { name: 'Computer difficulty' })).toHaveValue('2');
  await expect(dialog.locator('[data-side-choice="O"]')).toHaveAttribute('aria-pressed', 'true');
});

test('unavailable preference storage leaves theme selection and game usable', async ({ page }) => {
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => { throw new DOMException('Blocked', 'SecurityError'); };
    Storage.prototype.setItem = () => { throw new DOMException('Blocked', 'SecurityError'); };
    Storage.prototype.removeItem = () => { throw new DOMException('Blocked', 'SecurityError'); };
  });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  const theme = page.locator('[data-action="theme"]');
  await expect(theme).toHaveText('Auto');
  await theme.click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await theme.click();
  await theme.click();
  await expect(theme).toHaveText('Auto');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.locator('.cell:enabled').first().click();
  await expect(page.locator('.turn-title')).toHaveText('O to play');
});
