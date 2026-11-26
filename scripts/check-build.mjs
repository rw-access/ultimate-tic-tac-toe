import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { chromium, devices } from '@playwright/test';
import { prepareParityProbe, checkWorkerParity } from './production-parity.mjs';

// Serve only beneath a prefix to catch accidental absolute asset URLs.
const root = resolve('dist/play/ultimate-tic-tac-toe');
const prefix = '/play/ultimate-tic-tac-toe/';
const MAX_ONNX_MODEL_BYTES = 1_500_000;
const modelBytes = await readFile(resolve(root, 'models/policy-value.onnx'));
assert.ok(modelBytes.byteLength <= MAX_ONNX_MODEL_BYTES,
  `policy/value ONNX is ${modelBytes.byteLength} bytes; maximum is ${MAX_ONNX_MODEL_BYTES}`);
const parity = await prepareParityProbe(root, modelBytes, process.env.MODEL_PARITY_FIXTURE
  ?? new URL('../training/parity.json', import.meta.url));
const contentTypes = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm' };
const served = new Set();
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (!pathname.startsWith(prefix)) throw new Error('Outside site prefix');
    if (pathname === prefix + parity.entry.fileName) {
      response.writeHead(200, { 'Content-Type': 'text/javascript' });
      response.end(parity.entry.code);
      return;
    }
    const file = resolve(root, decodeURIComponent(pathname.slice(prefix.length)) || 'index.html');
    if (!file.startsWith(root + sep)) throw new Error('Outside build');
    const data = await readFile(file);
    served.add(pathname);
    response.writeHead(200, { 'Content-Type': contentTypes[extname(file)] ?? 'application/octet-stream' });
    response.end(data);
  } catch {
    response.writeHead(404);
    response.end('Not found');
  }
});
async function startHardAsX(page) {
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  const setup = page.getByRole('dialog', { name: 'New game' });
  await setup.getByRole('button', { name: 'Play vs computer' }).click();
  await setup.getByRole('slider', { name: 'Computer difficulty' }).press('End');
  await setup.getByRole('button', { name: 'X', exact: true }).click();
  await setup.getByRole('button', { name: 'Start game' }).click();
}

await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch();
  const address = server.address();
  const url = `http://127.0.0.1:${address.port}${prefix}`;
  for (const device of ['Desktop Chrome', 'iPhone 13']) {
    const { defaultBrowserType: _, ...options } = devices[device];
    const context = await browser.newContext(options);
    const page = await context.newPage();
    const errors = [];
    const requests = [];
    page.on('pageerror', error => errors.push(error.message));
    context.on('request', request => requests.push(request.url()));
    served.clear();
    const navigation = await page.goto(url);
    assert.equal(navigation?.status(), 200, 'Production index must load at the site prefix');
    await page.getByRole('button', { name: 'New game', exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Hint', exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Analysis', exact: true }).count(), 0);
    await page.getByRole('button', { name: 'How to play' }).click();
    assert.equal(await page.locator('#how-to-dialog .rules-list li').count(), 5);
    assert.equal(await page.locator('#how-to-dialog a').getAttribute('href'), 'https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe');
    await page.getByRole('button', { name: 'Got it' }).click();
    // Complete a two-player game using only the cells the UI allows. Full
    // analysis (advantage and alternatives) is now reached through game review.
    await page.evaluate(() => {
      for (let ply = 0; ply < 81; ply++) {
        const cell = window.document.querySelector('.cell:enabled');
        if (!cell) break;
        cell.click();
      }
    });
    await page.getByRole('button', { name: 'Review game' }).click();
    const review = page.getByRole('region', { name: 'Game review' });
    await review.getByRole('button', { name: 'Go to starting position' }).click();
    const openingMeter = review.getByRole('meter', { name: 'Position advantage' });
    await openingMeter.waitFor({ timeout: 25000 });
    const opening = parity.fixture.cases.find(item => item.name === 'initial');
    assert.ok(opening, 'Parity fixture must include the opening position');
    const openingReading = await openingMeter.getAttribute('aria-valuenow');
    assert.ok(openingReading !== null && Math.abs(Number(openingReading) + 100 * opening.value) <= 0.51,
      'Production game review must display the native opening value rounded to its meter scale');
    assert.match(await review.locator('.review-evaluation .analysis-note').innerText(), /Model estimate/);
    const lines = review.getByRole('region', { name: 'Best move and lines' });
    await lines.locator('.review-line').filter({ hasText: 'Best' }).waitFor({ timeout: 25000 });
    await lines.locator('.review-line').filter({ hasText: 'Alt 1' }).waitFor({ timeout: 25000 });
    await page.getByRole('button', { name: 'Exit review' }).click();
    await review.waitFor({ state: 'detached' });

    await startHardAsX(page);
    await page.getByRole('button', { name: 'Hint', exact: true }).click();
    await page.locator('.cell-suggested:enabled').waitFor({ timeout: 25000 });
    assert.equal(await page.locator('.cell-suggested').count(), 1);
    assert.equal(await page.locator('.analysis-panel, #analysis-dialog').count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Analysis', exact: true }).count(), 0);
    const elapsed = [];
    for (let round = 1; round <= 3; round++) {
      const start = performance.now();
      await page.locator('.cell:enabled').first().click();
      await page.locator('[data-computer-source="model"]').waitFor({ timeout: 25000 });
      elapsed.push(Math.round(performance.now() - start));
      assert.equal(await page.locator('.cell-o').count(), round);
      assert.equal(await page.locator('.cell-x').count(), round);
    }
    if (device === 'Desktop Chrome') {
      await checkWorkerParity(page, url, parity);
      for (const height of [720, 801, 850, 900]) {
        await page.setViewportSize({ width: 1280, height });
        const viewport = await page.evaluate(() => ({ scrollHeight: document.documentElement.scrollHeight, innerHeight }));
        assert.ok(viewport.scrollHeight <= viewport.innerHeight, `${device} ${height}px: vertical overflow ${JSON.stringify(viewport)}`);
      }
    }
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${device}: horizontal overflow`);
    const layout = await page.evaluate(() => {
      const box = (selector) => {
        const element = document.querySelector(selector);
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        return { top: Math.round(rect.top), bottom: Math.round(rect.bottom), height: Math.round(rect.height), width: Math.round(rect.width) };
      };
      return {
        scrollHeight: document.documentElement.scrollHeight,
        innerHeight,
        innerWidth,
        shell: box('.page-shell'), settings: box('.game-settings'), toolbar: box('.toolbar'),
        status: box('.computer-status'), board: box('.board-frame'), footer: box('.under-board'),
      };
    });
    if (device === 'Desktop Chrome') {
      assert.ok(layout.scrollHeight <= layout.innerHeight, `${device}: vertical overflow ${JSON.stringify(layout)}`);
    } else {
      assert.ok(layout.board.width >= layout.innerWidth - 32, `${device}: board is too narrow ${JSON.stringify(layout)}`);
    }
    assert.deepEqual(errors, []);
    assert.ok(served.has(prefix + 'models/policy-value.onnx'), 'Production worker must fetch the bundled model');
    assert.ok([...served].some(path => path.endsWith('.wasm')), 'Production worker must fetch the WASM runtime');
    const networkRequests = requests.map(value => new URL(value)).filter(request => ['http:', 'https:'].includes(request.protocol));
    assert.ok(networkRequests.every(request => request.origin === new URL(url).origin && request.pathname.startsWith(prefix)),
      'All network requests must stay within the production prefix');
    console.log(`${device}: Hint, game review and production model play beneath ${prefix} passed; move wall times ${elapsed.join(', ')} ms`);
    await context.close();
  }

  // Exercise worker recovery with a fresh session, so successful model loads
  // from the preceding devices cannot satisfy the fallback assertions.
  const fallback = await browser.newContext();
  let rejected = 0;
  await fallback.route(url + 'models/policy-value.onnx', route => {
    rejected++;
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: 'invalid ONNX data' });
  });
  const page = await fallback.newPage();
  await page.goto(url);
  await startHardAsX(page);
  await page.locator('[data-board="4"][data-cell="2"]').click();
  await page.locator('[data-computer-source="search"]').waitFor({ timeout: 25000 });
  assert.ok(rejected > 0, 'Fallback must attempt the corrupt model');
  assert.equal(await page.locator('[data-mini-board="2"] .cell-o').count(), 1);
  assert.equal(await page.locator('.turn-title').textContent(), 'X to play');
  await fallback.close();
  console.log('Corrupt production model recovered with a legal search reply');
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
