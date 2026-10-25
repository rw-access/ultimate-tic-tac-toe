import { test, expect } from '@playwright/test';

async function chooseComputer(page: import('@playwright/test').Page, difficulty = 'medium'): Promise<void> {
  await page.getByRole('button', { name: 'New game' }).click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await dialog.getByRole('button', { name: 'Play vs computer' }).click();
  const slider = page.getByRole('slider', { name: 'Computer difficulty' });
  if (difficulty === 'easy') await slider.press('Home');
  else if (difficulty === 'hard') await slider.press('End');
  await dialog.getByRole('button', { name: 'Start game' }).click();
}

test('new-game menu keeps mode, difficulty, and side choices together', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Two players' })).toHaveCount(0);
  await expect(page.getByRole('slider', { name: 'Computer difficulty' })).toHaveCount(0);

  await page.getByRole('button', { name: 'New game' }).click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  const twoPlayers = dialog.getByRole('button', { name: 'Two players' });
  const computer = dialog.getByRole('button', { name: 'Play vs computer' });
  await expect(twoPlayers).toHaveAttribute('aria-pressed', 'true');
  await computer.click();
  const slider = page.getByRole('slider', { name: 'Computer difficulty' });
  await expect(computer).toHaveAttribute('aria-pressed', 'true');
  await expect(slider).toHaveValue('1');
  await expect(slider).toHaveAttribute('aria-valuetext', 'Medium');
  await expect(page.locator('#computer-difficulty-help')).toHaveText('More search before each move.');
  await expect(dialog.getByRole('group', { name: 'Choose your side' }).getByRole('button')).toHaveCount(3);

  await slider.press('End');
  await expect(page.getByRole('slider', { name: 'Computer difficulty' })).toHaveValue('2');
  await expect(page.getByRole('slider', { name: 'Computer difficulty' })).toHaveAttribute('aria-valuetext', 'Hard');
  await expect(page.getByRole('slider', { name: 'Computer difficulty' })).toBeFocused();
  await twoPlayers.click();
  await expect(page.getByRole('slider', { name: 'Computer difficulty' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'New game' })).toHaveCount(1);
  await dialog.getByRole('button', { name: 'Cancel' }).click();
});

test('computer answers with the ONNX model and makes a legal O move', async ({ page }) => {
  await page.goto('/');
  await chooseComputer(page);
  await page.locator('[data-board="4"][data-cell="4"]').click();

  const status = page.locator('.computer-status');
  await expect(status).toHaveAttribute('data-computer-source', 'model', { timeout: 25000 });
  await expect(status).toHaveText('Your turn · you play X.');
  await expect(page.locator('.cell-x')).toHaveCount(1);
  await expect(page.locator('.cell-o')).toHaveCount(1);
  await expect(page.locator('.turn-title')).toHaveText('X to play');
});

test('undo removes the complete human and computer round', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({ status: 404, body: 'missing model' }));
  await page.goto('/');
  await chooseComputer(page);
  await page.locator('[data-board="4"][data-cell="4"]').click();

  await expect(page.locator('.computer-status')).toHaveAttribute('data-computer-source', 'search');
  await expect(page.locator('.cell-o')).toHaveCount(1);
  await page.getByRole('button', { name: 'Undo' }).click();
  await expect(page.locator('.turn-title')).toHaveText('X to play');
  await expect(page.locator('.cell-x, .cell-o')).toHaveCount(0);
  await expect(page.locator('.cell:enabled')).toHaveCount(81);
});

test('starting a computer game from the picker resets the old two-player game', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({ status: 404, body: 'missing model' }));
  await page.goto('/');
  await page.locator('[data-board="4"][data-cell="4"]').click();
  await page.locator('[data-board="4"][data-cell="0"]').click();
  await expect(page.locator('.cell-x')).toHaveCount(1);
  await expect(page.locator('.cell-o')).toHaveCount(1);

  await chooseComputer(page);
  await expect(page.locator('.turn-title')).toHaveText('X to play');
  await expect(page.locator('.cell-x, .cell-o')).toHaveCount(0);
  await page.locator('[data-board="4"][data-cell="4"]').click();
  await expect(page.locator('.computer-status')).toHaveAttribute('data-computer-source', 'search');
  await expect(page.locator('.cell-x')).toHaveCount(1);
  await expect(page.locator('.cell-o')).toHaveCount(1);
  await page.getByRole('button', { name: 'Undo' }).click();
  await expect(page.locator('.cell-x, .cell-o')).toHaveCount(0);
});

test('undo and new game cancel a pending computer reply', async ({ page }) => {
  const pendingReplies: (() => Promise<void>)[] = [];
  await page.route('**/models/policy-value.onnx', route => new Promise<void>(resolve => {
    pendingReplies.push(async () => {
      try {
        await route.fulfill({ status: 404, body: 'missing model' });
      } catch {
        // The client should have aborted this request when its worker was cancelled.
      }
      resolve();
    });
  }));
  await page.goto('/');
  await chooseComputer(page, 'hard');
  await page.locator('[data-board="4"][data-cell="4"]').click();
  await expect(page.locator('.turn-title')).toHaveText('Computer thinking…');
  await expect.poll(() => pendingReplies.length).toBeGreaterThan(0);

  await page.getByRole('button', { name: 'Undo' }).click();
  await Promise.all(pendingReplies.splice(0).map(release => release()));
  await expect(page.locator('.cell-x, .cell-o')).toHaveCount(0);
  await expect(page.locator('.turn-title')).toHaveText('X to play');

  await page.locator('[data-board="4"][data-cell="4"]').click();
  await expect(page.locator('.turn-title')).toHaveText('Computer thinking…');
  await expect.poll(() => pendingReplies.length).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'New game' }).first().click();
  await page.locator('[data-action="start-new-game"]').click();
  await Promise.all(pendingReplies.splice(0).map(release => release()));
  await expect(page.locator('.turn-title')).toHaveText('X to play');
  await expect(page.locator('.cell-x, .cell-o')).toHaveCount(0);
});

test('opponent preferences persist and a restored O turn receives a legal fallback reply', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', async route => {
    await new Promise(resolve => setTimeout(resolve, 700));
    try {
      await route.fulfill({ status: 404, body: 'missing model' });
    } catch {
      // The first request is expected to be interrupted by the reload below.
    }
  });
  await page.goto('/');
  await chooseComputer(page, 'hard');
  await page.locator('.cell:enabled').first().click();
  await expect(page.locator('.turn-title')).toHaveText('Computer thinking…');
  await page.reload();

  await page.getByRole('button', { name: 'New game' }).click();
  const dialog = page.getByRole('dialog', { name: 'New game' });
  await expect(dialog.getByRole('button', { name: 'Play vs computer' })).toHaveAttribute('aria-pressed', 'true');
  await expect(dialog.getByRole('slider', { name: 'Computer difficulty' })).toHaveValue('2');
  await expect(dialog.getByRole('button', { name: 'X', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await dialog.getByRole('button', { name: 'Cancel' }).click();
  await expect(page.locator('.cell-x')).toHaveCount(1);
  await expect(page.locator('.computer-status')).toHaveAttribute('data-computer-source', 'search', { timeout: 10000 });
  await expect(page.locator('.computer-status')).toHaveText('You play X · computer uses basic search.');
  await expect(page.locator('.cell-o')).toHaveCount(1);
  await expect(page.locator('.turn-title')).toHaveText('X to play');
});
