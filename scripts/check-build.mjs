import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { chromium, devices } from '@playwright/test';

// Serve only beneath a prefix to catch accidental absolute asset URLs.
const root = resolve('dist/play/ultimate-tic-tac-toe');
const prefix = '/play/ultimate-tic-tac-toe/';
const MAX_ONNX_MODEL_BYTES = 1_500_000;
const modelBytes = await readFile(resolve(root, 'models/policy-value.onnx'));
assert.ok(modelBytes.byteLength <= MAX_ONNX_MODEL_BYTES,
  `policy/value ONNX is ${modelBytes.byteLength} bytes; maximum is ${MAX_ONNX_MODEL_BYTES}`);
const contentTypes = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm' };
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (!pathname.startsWith(prefix)) throw new Error('Outside site prefix');
    const file = resolve(root, decodeURIComponent(pathname.slice(prefix.length)) || 'index.html');
    if (!file.startsWith(root + sep)) throw new Error('Outside build');
    const data = await readFile(file);
    response.writeHead(200, { 'Content-Type': contentTypes[extname(file)] ?? 'application/octet-stream' });
    response.end(data);
  } catch {
    response.writeHead(404);
    response.end('Not found');
  }
});
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
    page.on('request', request => requests.push(request.url()));
    await page.goto(url);
    await page.getByRole('button', { name: 'Hint', exact: true }).click();
    await page.locator('.cell-suggested').waitFor({ timeout: 25000 });
    assert.equal(await page.locator('.analysis-panel').count(), 0);
    await page.getByRole('button', { name: 'Analysis', exact: true }).click();
    await page.locator('#analysis-dialog .advantage-track').waitFor({ timeout: 25000 });
    assert.equal(await page.locator('#analysis-dialog .candidate-lines .candidate-line').count(), 1);
    assert.equal(await page.locator('#analysis-dialog .alternative-lines').count(), 1);
    await page.getByRole('button', { name: 'Done' }).click();
    await page.waitForSelector('#analysis-dialog', { state: 'detached' });
    await page.getByRole('button', { name: 'How to play' }).click();
    assert.equal(await page.locator('#how-to-dialog .rules-list li').count(), 5);
    assert.equal(await page.locator('#how-to-dialog a').getAttribute('href'), 'https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe');
    await page.getByRole('button', { name: 'Got it' }).click();
    await page.getByRole('button', { name: 'New game' }).click();
    const newGame = page.getByRole('dialog', { name: 'New game' });
    await newGame.getByRole('button', { name: 'Play vs computer' }).click();
    await newGame.getByRole('slider', { name: 'Computer difficulty' }).press('End');
    await newGame.getByRole('button', { name: 'Start game' }).click();
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
    assert.ok(requests.some(request => request.endsWith('/models/policy-value.onnx')));
    assert.ok(requests.some(request => request.endsWith('.wasm')));
    assert.ok(requests.every(request => new URL(request).pathname.startsWith(prefix)));
    console.log(`${device}: separate Hint/Analysis and production model play beneath ${prefix} passed; move wall times ${elapsed.join(', ')} ms`);
    await context.close();
  }
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
