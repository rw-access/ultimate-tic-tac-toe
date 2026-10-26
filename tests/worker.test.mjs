import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const CODE_PATTERN = /^[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{8}$/;

async function availablePort() {
  const server = createServer();
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  await new Promise(resolveClose => server.close(resolveClose));
  assert.ok(address && typeof address !== 'string');
  return address.port;
}

function nextMessage(socket, predicate = () => true) {
  return new Promise((resolveMessage, reject) => {
    const timer = setTimeout(() => {
      socket.removeEventListener('message', onMessage);
      reject(new Error('Timed out waiting for a room WebSocket message.'));
    }, 5000);
    function onMessage(event) {
      const value = JSON.parse(String(event.data));
      if (!predicate(value)) return;
      clearTimeout(timer);
      socket.removeEventListener('message', onMessage);
      resolveMessage(value);
    }
    socket.addEventListener('message', onMessage);
  });
}

async function connectSeat(url, token) {
  const socket = new WebSocket(url);
  await new Promise((resolveOpen, reject) => {
    socket.addEventListener('open', resolveOpen, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  const snapshot = nextMessage(socket, message => message.type === 'snapshot');
  socket.send(JSON.stringify({ type: 'hello', token }));
  return { socket, snapshot: await snapshot };
}

test('local Worker serves the app and enforces real two-seat rooms over WebSockets', { timeout: 60_000 }, async t => {
  const port = await availablePort();
  const persistence = mkdtempSync(join(tmpdir(), 'ultimate-ttt-worker-test-'));
  const wrangler = join(root, 'node_modules', 'wrangler', 'bin', 'wrangler.js');
  const child = spawn(process.execPath, [wrangler, 'dev', '--local', '--ip', '127.0.0.1', '--port', String(port), '--persist-to', persistence], {
    cwd: root,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '';
  child.stdout.setEncoding('utf8').on('data', chunk => { logs += chunk; });
  child.stderr.setEncoding('utf8').on('data', chunk => { logs += chunk; });
  const stop = () => new Promise(resolveStop => {
    if (child.exitCode !== null) return resolveStop();
    child.once('exit', resolveStop);
    child.kill('SIGTERM');
  });
  t.after(async () => {
    await stop();
    rmSync(persistence, { recursive: true, force: true });
  });

  const base = `http://127.0.0.1:${port}`;
  let ready = false;
  const deadline = Date.now() + 25_000;
  while (Date.now() < deadline && child.exitCode === null) {
    try {
      const response = await fetch(`${base}/api/health`);
      if (response.ok && (await response.json()).ok === true) {
        ready = true;
        break;
      }
    } catch { /* Wrangler is still starting. */ }
    await new Promise(resolveDelay => setTimeout(resolveDelay, 150));
  }
  assert.equal(ready, true, `Worker did not start. ${logs}`);

  const page = await fetch(`${base}/`);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /<div id="app"><\/div>/);

  const createResponse = await fetch(`${base}/api/rooms`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(createResponse.status, 201);
  const host = await createResponse.json();
  assert.match(host.code, CODE_PATTERN);
  assert.equal(host.seat, 'X');
  assert.equal(host.room.players, 1);
  assert.equal(JSON.stringify(host.room).includes(host.token), false);

  const joinResponse = await fetch(`${base}/api/rooms/${host.code}/join`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(joinResponse.status, 200);
  const guest = await joinResponse.json();
  assert.equal(guest.seat, 'O');
  assert.equal(guest.room.players, 2);
  const thirdSeat = await fetch(`${base}/api/rooms/${host.code}/join`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(thirdSeat.status, 409);

  const wsUrl = `ws://127.0.0.1:${port}/api/rooms/${host.code}/socket`;
  const hostClient = await connectSeat(wsUrl, host.token);
  assert.equal(hostClient.snapshot.room.version, 0);
  const guestClient = await connectSeat(wsUrl, guest.token);
  assert.equal(guestClient.snapshot.room.players, 2);

  const afterHostMove = nextMessage(guestClient.socket, message => message.type === 'snapshot' && message.room.version === 1);
  hostClient.socket.send(JSON.stringify({ type: 'move', board: 0, cell: 0, version: 0 }));
  const hostMove = await afterHostMove;
  assert.equal(hostMove.room.state.boards[0][0], 'X');

  const rejectedMove = nextMessage(hostClient.socket, message => message.type === 'error');
  hostClient.socket.send(JSON.stringify({ type: 'move', board: 0, cell: 1, version: 1 }));
  assert.match((await rejectedMove).message, /turn/i);

  const afterGuestMove = nextMessage(hostClient.socket, message => message.type === 'snapshot' && message.room.version === 2);
  guestClient.socket.send(JSON.stringify({ type: 'move', board: 0, cell: 1, version: 1 }));
  const guestMove = await afterGuestMove;
  assert.equal(guestMove.room.state.boards[0][1], 'O');
  assert.deepEqual(guestMove.room.moves, [0, 1]);

  hostClient.socket.close();
  guestClient.socket.close();
});
