import test from 'node:test';
import assert from 'node:assert/strict';
import { APP_BASE_PATH, appPath, stripAppBasePath } from '../src/app-path.ts';

test('online URLs stay under the game prefix in production and at root in local Vite dev', () => {
  assert.equal(APP_BASE_PATH, '/play/ultimate-tic-tac-toe');
  assert.equal(appPath(`${APP_BASE_PATH}/`, '/api/rooms'), `${APP_BASE_PATH}/api/rooms`);
  assert.equal(appPath('/', '/api/rooms'), '/api/rooms');
  assert.equal(stripAppBasePath(`${APP_BASE_PATH}/api/rooms`), '/api/rooms');
  assert.equal(stripAppBasePath('/elsewhere/api/rooms'), null);
});

test('existing invites keep their legacy API prefix and path matching respects boundaries', () => {
  const legacy = '/ultimate-tic-tac-toe';
  assert.equal(appPath(`${legacy}/`, '/api/rooms'), `${legacy}/api/rooms`);
  assert.equal(stripAppBasePath(`${legacy}/api/rooms`), '/api/rooms');
  for (const base of [APP_BASE_PATH, legacy]) {
    assert.equal(stripAppBasePath(base), '/');
    assert.equal(stripAppBasePath(`${base}/`), '/');
    assert.equal(stripAppBasePath(`${base}-other/api/rooms`), null);
    assert.equal(appPath(`${base}/`, 'api/rooms'), `${base}/api/rooms`);
  }
});
