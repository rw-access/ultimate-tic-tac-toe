import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createGame, legalBoards, play, type GameState } from '../../src/game';

test('Hint gives one move when playing the computer', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({ status: 404, body: 'use search fallback' }));
  await page.goto('/');
  await page.getByRole('button', { name: 'New game' }).click();
  const setup = page.getByRole('dialog', { name: 'New game' });
  await setup.getByRole('button', { name: 'Play vs computer' }).click();
  await setup.getByRole('button', { name: 'X', exact: true }).click();
  await setup.getByRole('button', { name: 'Start game' }).click();

  await expect(page.locator('.analysis-panel')).toHaveCount(0);
  await expect(page.locator('#analysis-dialog')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Hint', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: 'Hint', exact: true }).click();
  await expect(page.locator('.cell-suggested')).toHaveCount(1, { timeout: 20000 });
  await expect(page.locator('.analysis-panel')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Hint shown' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);
});

test('live Hint and Analysis stay unavailable in two-player mode', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Hint', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);

  await page.locator('[data-board="4"][data-cell="4"]').click();
  await expect(page.getByRole('button', { name: 'Hint', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);
});

test('game review layout adapts across desktop and phone widths', async ({ page }) => {
  let state = createGame();
  const history: GameState[] = [];
  let seed = 8917;
  while (!state.winner && state.moves < 81) {
    const moves = legalBoards(state).flatMap(board => state.boards[board].flatMap((mark, cell) =>
      mark === null ? [board * 9 + cell] : [],
    ));
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    history.push(state);
    const move = moves[seed % moves.length];
    state = play(state, Math.floor(move / 9), move % 9);
  }

  await page.goto('/');
  await page.evaluate(saved => localStorage.setItem('ultimate-tic-tac-toe:v1', JSON.stringify({ version: 1, state: saved.state, undo: saved.history })), { state, history });
  await page.reload();
  await page.getByRole('button', { name: 'Review game' }).click();
  const review = page.getByRole('region', { name: 'Game review' });
  await expect(review).toBeVisible();

  for (const viewport of [
    { width: 1280, height: 720, sideBySide: true, minBoard: 560, minPanel: 650 },
    { width: 1440, height: 900, sideBySide: true, minBoard: 600, minPanel: 700 },
    { width: 1920, height: 1080, sideBySide: true, minBoard: 700, minPanel: 850 },
    { width: 1024, height: 768, sideBySide: false, minBoard: 440, minPanel: 440 },
    { width: 375, height: 844, sideBySide: false, minBoard: 347, minPanel: 347 },
    { width: 390, height: 844, sideBySide: false, minBoard: 362, minPanel: 362 },
  ]) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    const layout = await page.evaluate(() => {
      const board = document.querySelector('.board-frame')!.getBoundingClientRect();
      const panel = document.querySelector('.review-panel')!.getBoundingClientRect();
      return {
        width: innerWidth,
        scrollWidth: document.documentElement.scrollWidth,
        board: { left: board.left, right: board.right, top: board.top, bottom: board.bottom, width: board.width, height: board.height },
        panel: { left: panel.left, right: panel.right, top: panel.top, width: panel.width },
      };
    });
    expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width + 1);
    expect(layout.board.left).toBeGreaterThanOrEqual(-1);
    expect(layout.board.right).toBeLessThanOrEqual(layout.width + 1);
    expect(layout.panel.left).toBeGreaterThanOrEqual(-1);
    expect(layout.panel.right).toBeLessThanOrEqual(layout.width + 1);
    expect(layout.board.width).toBeGreaterThanOrEqual(viewport.minBoard);
    expect(layout.panel.width).toBeGreaterThanOrEqual(viewport.minPanel);
    expect(Math.abs(layout.board.width - layout.board.height)).toBeLessThanOrEqual(1);
    if (viewport.sideBySide) expect(layout.panel.left).toBeGreaterThanOrEqual(layout.board.right - 1);
    else expect(layout.panel.top).toBeGreaterThan(layout.board.bottom);
  }
});

test('a completed game can be reviewed move by move', async ({ page }) => {
  test.setTimeout(60000);
  let state = createGame();
  const history: GameState[] = [];
  let seed = 8917;
  while (!state.winner && state.moves < 81) {
    const moves = legalBoards(state).flatMap(board => state.boards[board].flatMap((mark, cell) =>
      mark === null ? [board * 9 + cell] : [],
    ));
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    history.push(state);
    const move = moves[seed % moves.length];
    state = play(state, Math.floor(move / 9), move % 9);
  }
  expect(state.winner).not.toBeNull();

  await page.goto('/');
  await expect(page).toHaveTitle('Ultimate Tic-Tac-Toe');
  await expect(page.getByRole('heading', { name: 'Ultimate Tic-Tac-Toe' })).toBeVisible();
  await page.evaluate(saved => localStorage.setItem('ultimate-tic-tac-toe:v1', JSON.stringify({ version: 1, state: saved.state, undo: saved.history })), { state, history });
  await page.reload();
  await expect(page.locator('.turn-title')).not.toHaveText(/to play/);
  const savedGame = await page.evaluate(() => localStorage.getItem('ultimate-tic-tac-toe:v1'));
  await page.getByRole('button', { name: 'Review game' }).click();
  const review = page.getByRole('region', { name: 'Game review' });
  await expect(review).toBeVisible();
  await expect(page.locator('#analysis-dialog')).toHaveCount(0);
  const initialLayout = await page.evaluate(() => {
    const board = document.querySelector('.board-frame')!.getBoundingClientRect();
    const panel = document.querySelector('.review-panel')!.getBoundingClientRect();
    return { width: innerWidth, height: innerHeight, boardRight: board.right, boardBottom: board.bottom, boardWidth: board.width, panelLeft: panel.left, panelTop: panel.top, panelWidth: panel.width };
  });
  if (initialLayout.width >= 1280 && initialLayout.height >= 620) {
    expect(initialLayout.panelLeft).toBeGreaterThanOrEqual(initialLayout.boardRight - 1);
    expect(initialLayout.boardWidth).toBeGreaterThanOrEqual(560);
    expect(initialLayout.panelWidth).toBeGreaterThanOrEqual(650);
  } else {
    expect(initialLayout.panelTop).toBeGreaterThan(initialLayout.boardBottom);
  }
  const turns = review.locator('.review-move-pair');
  await expect(turns).toHaveCount(Math.ceil(history.length / 2));
  await expect(turns.first().locator('[data-player="X"]')).toHaveCount(1);
  await expect(turns.first().locator('[data-player="O"]')).toHaveCount(1);
  const pairedMoveOrder = await turns.first().evaluate(turn => {
    const x = turn.querySelector('[data-player="X"]')!.getBoundingClientRect();
    const o = turn.querySelector('[data-player="O"]')!.getBoundingClientRect();
    return x.left < o.left;
  });
  expect(pairedMoveOrder).toBe(true);
  await expect(review.getByRole('group', { name: 'Replay and move navigation' })).toBeVisible();
  await expect(review.locator('.review-move-start')).toHaveCount(0);
  await expect(review.getByRole('button', { name: 'Play replay' })).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await review.getByRole('button', { name: 'Export game notation' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('ultimate-tic-tac-toe-game.txt');
  const downloadedPath = await download.path();
  expect(downloadedPath).not.toBeNull();
  const notation = await readFile(downloadedPath!, 'utf8');
  expect(notation).toContain('Ultimate Tic-Tac-Toe');
  expect(notation).toContain(`Result: ${state.winner === 'draw' ? 'Draw' : `${state.winner} won`} · ${history.length} moves`);
  expect(notation).toContain('Coordinates: B<board>·<square>; board and square are numbered 1–9 row-major from the top-left.');
  const numberedMoves = notation.split('\n').filter(line => /^\d+\.\s/.test(line));
  expect(numberedMoves).toHaveLength(Math.ceil(history.length / 2));
  expect(numberedMoves[0]).toMatch(/^1\. X B[1-9]·[1-9] O B[1-9]·[1-9]$/);
  const chips = review.locator('.review-move-chip');
  await expect(chips).toHaveCount(history.length);
  const currentChipVisible = await review.locator('.review-move-chip.is-current').evaluate(button => {
    const strip = button.closest('.review-move-strip')!.getBoundingClientRect();
    const move = button.getBoundingClientRect();
    return move.top >= strip.top && move.bottom <= strip.bottom;
  });
  expect(currentChipVisible).toBe(true);
  const slider = review.getByRole('slider', { name: 'Review move' });
  await expect(slider).toHaveValue(String(history.length - 1));
  await expect(review.locator('.review-outcome')).toContainText(/(X|O) won · \d+ moves|Draw · \d+ moves/);
  await expect(review.getByRole('region', { name: 'Critical moves' })).toBeVisible();
  await expect(review.locator(`.review-move-chip[data-review-position="${history.length}"]`)).toHaveClass(/is-future/);
  await expect(review.locator('.review-graph-future')).toHaveCount(1);

  await review.getByRole('button', { name: 'Go to starting position' }).click();
  await expect(slider).toHaveValue('0');
  await expect(review.locator('.review-move-chip.is-current')).toHaveCount(0);
  await expect.poll(() => review.locator('.review-move-strip').evaluate(element => element.scrollTop)).toBe(0);
  await expect(page.locator('.cell-review-current')).toHaveCount(0);
  await expect(review.locator('.advantage-track[role="meter"]')).toBeVisible({ timeout: 20000 });
  await expect(review.locator('.review-graph-point[data-review-position="0"]')).toHaveCount(1, { timeout: 20000 });
  await expect(review.locator('.review-graph-future')).toHaveAttribute('width', /[1-9]/);
  await expect(review.locator('.review-lines .review-line')).toHaveCount(2, { timeout: 20000 });

  await review.getByRole('button', { name: 'Play replay' }).click();
  await expect(slider).toHaveValue('1', { timeout: 3000 });
  await review.getByRole('button', { name: /Playback speed 1×/ }).click();
  await expect(review.getByRole('button', { name: /Playback speed 2×/ })).toHaveText('2×');
  await review.locator('.review-move-chip[data-review-position="2"]').click();
  await expect(slider).toHaveValue('2');
  await expect(review.getByRole('button', { name: 'Play replay' })).toBeVisible();
  await review.locator(`.review-move-chip[data-review-position="${history.length - 1}"]`).click();
  await review.getByRole('button', { name: 'Play replay' }).click();
  await expect(slider).toHaveValue(String(history.length), { timeout: 3000 });
  await expect(review.getByRole('button', { name: 'Play replay' })).toBeVisible();

  await slider.evaluate((element, value) => {
    const range = element as HTMLInputElement;
    range.value = String(value);
    range.dispatchEvent(new Event('change', { bubbles: true }));
  }, Math.min(2, history.length));
  await expect(slider).toHaveValue(String(Math.min(2, history.length)));
  await expect(page.locator('.cell-review-current')).toHaveCount(1);
  const selectedMoveName = await page.locator('.cell-review-current').getAttribute('aria-label');
  expect(selectedMoveName).toContain('reviewed move');

  await expect.poll(async () => review.locator('.review-summary').textContent(), { timeout: 30000 }).toContain('moves reviewed');
  await expect(review.locator('.review-graph-point')).toHaveCount(history.length + 1);
  await expect(chips.locator('.review-grade')).toHaveCount(history.length);
  await expect(chips.locator('.review-grade svg')).toHaveCount(history.length);
  const secondGraphPoint = review.locator('.review-graph-point[data-review-position="1"]');
  await secondGraphPoint.click();
  await expect(slider).toHaveValue('1');
  await expect(secondGraphPoint).toBeFocused();
  const criticalMove = review.locator('.review-critical-move').first();
  if (await criticalMove.count()) {
    const comparisonCount = await criticalMove.locator('.review-move-choice').count();
    expect([1, 2]).toContain(comparisonCount);
    await expect(criticalMove.locator('.review-move-choice.is-best')).toContainText('Best');
    const criticalText = await criticalMove.innerText();
    expect(criticalText).not.toMatch(/Evaluation fell|toward|Top model pick/);
    const target = await criticalMove.getAttribute('data-review-position');
    await criticalMove.click();
    await expect(slider).toHaveValue(target!);
  }
  if (initialLayout.width >= 1280 && initialLayout.height >= 620) {
    await page.setViewportSize({ width: 1024, height: 768 });
    const stackedLayout = await page.evaluate(() => {
      const board = document.querySelector('.board-frame')!.getBoundingClientRect();
      const panel = document.querySelector('.review-panel')!.getBoundingClientRect();
      return { boardBottom: board.bottom, panelTop: panel.top, boardWidth: board.width, panelWidth: panel.width };
    });
    expect(stackedLayout.panelTop).toBeGreaterThan(stackedLayout.boardBottom);
    expect(stackedLayout.boardWidth).toBeGreaterThan(440);
    expect(stackedLayout.panelWidth).toBeGreaterThan(440);

    await page.setViewportSize({ width: 1440, height: 900 });
    const wideLayout = await page.evaluate(() => {
      const board = document.querySelector('.board-frame')!.getBoundingClientRect();
      const panel = document.querySelector('.review-panel')!.getBoundingClientRect();
      return { boardRight: board.right, boardWidth: board.width, panelLeft: panel.left, panelWidth: panel.width };
    });
    expect(wideLayout.panelLeft).toBeGreaterThanOrEqual(wideLayout.boardRight - 1);
    expect(wideLayout.boardWidth).toBeGreaterThan(600);
    expect(wideLayout.panelWidth).toBeGreaterThan(700);
  }
  await page.setViewportSize({ width: 320, height: 812 });
  const narrowExport = await review.locator('.review-export').evaluate(button => {
    const header = button.closest('.review-heading')!.getBoundingClientRect();
    const bounds = button.getBoundingClientRect();
    return { headerLeft: header.left, headerRight: header.right, buttonLeft: bounds.left, buttonRight: bounds.right };
  });
  expect(narrowExport.buttonLeft).toBeGreaterThanOrEqual(narrowExport.headerLeft - 1);
  expect(narrowExport.buttonRight).toBeLessThanOrEqual(narrowExport.headerRight + 1);
  const narrowNavigation = await review.locator('.review-navigation').evaluate(element => {
    const nav = element.getBoundingClientRect();
    const controls = element.querySelector('.review-step-buttons').getBoundingClientRect();
    return { navRight: nav.right, controlsRight: controls.right, scrollWidth: element.scrollWidth, clientWidth: element.clientWidth };
  });
  expect(narrowNavigation.controlsRight).toBeLessThanOrEqual(narrowNavigation.navRight + 1);
  expect(narrowNavigation.scrollWidth).toBeLessThanOrEqual(narrowNavigation.clientWidth + 1);
  const narrowTimeline = await review.locator('.review-move-strip').evaluate(element => ({
    overflowY: getComputedStyle(element).overflowY,
    scrollHeight: element.scrollHeight,
    clientHeight: element.clientHeight,
    scrollWidth: element.scrollWidth,
    clientWidth: element.clientWidth,
  }));
  expect(narrowTimeline.overflowY).toBe('auto');
  expect(narrowTimeline.scrollHeight).toBeGreaterThan(narrowTimeline.clientHeight);
  expect(narrowTimeline.scrollWidth).toBeLessThanOrEqual(narrowTimeline.clientWidth + 1);
  const narrowPage = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, width: innerWidth }));
  expect(narrowPage.scrollWidth).toBeLessThanOrEqual(narrowPage.width);
  await review.locator(`.review-graph-point[data-review-position="${history.length}"]`).click();
  await expect(slider).toHaveValue(String(history.length));
  await expect(review.locator('.review-lines')).toContainText('Game complete');
  await expect(page.locator('.cell-review-current')).toHaveCount(1);

  await page.getByRole('button', { name: 'Exit review' }).click();
  await expect(page.locator('.review-panel')).toHaveCount(0);
  await expect(page.locator('.turn-title')).not.toHaveText(/to play/);
  expect(await page.evaluate(() => localStorage.getItem('ultimate-tic-tac-toe:v1'))).toBe(savedGame);
});

test('Hint stays disabled for the computer opening and enables on the human O turn', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', async route => {
    await new Promise(resolve => setTimeout(resolve, 700));
    await route.fulfill({ status: 404, body: 'use search fallback' });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'New game' }).click();
  const setup = page.getByRole('dialog', { name: 'New game' });
  await setup.getByRole('button', { name: 'Play vs computer' }).click();
  await setup.getByRole('button', { name: 'O', exact: true }).click();
  await setup.getByRole('button', { name: 'Start game' }).click();

  const hint = page.getByRole('button', { name: 'Hint', exact: true });
  await expect(hint).toBeVisible();
  await expect(hint).toBeDisabled();
  await expect(page.locator('.cell-x')).toHaveCount(1, { timeout: 20000 });
  await expect(hint).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);
});

test('a new computer game can assign O to the player so the computer opens as X', async ({ page }) => {
  await page.route('**/models/policy-value.onnx', route => route.fulfill({ status: 404, body: 'missing model' }));
  await page.goto('/');
  await page.getByRole('button', { name: 'New game' }).click();

  const dialog = page.getByRole('dialog', { name: 'New game' });
  await dialog.getByRole('button', { name: 'Play vs computer' }).click();
  await expect(dialog.getByRole('group', { name: 'Choose your side' }).getByRole('button')).toHaveCount(3);
  await dialog.getByRole('button', { name: 'O', exact: true }).click();
  await dialog.getByRole('button', { name: 'Start game' }).click();

  await expect(page.locator('.computer-status')).toHaveAttribute('data-computer-source', 'search');
  await expect(page.locator('.computer-status')).toHaveText('You play O · computer uses basic search.');
  await expect(page.locator('.cell-x')).toHaveCount(1);
  await expect(page.locator('.cell-o')).toHaveCount(0);
  await expect(page.locator('.turn-title')).toHaveText('O to play');

  await page.locator('.cell:enabled').first().click();
  await expect(page.locator('.computer-status')).toHaveAttribute('data-computer-source', 'search');
  await expect(page.locator('.cell-o')).toHaveCount(1);
  await expect(page.locator('.cell-x')).toHaveCount(2);
});
