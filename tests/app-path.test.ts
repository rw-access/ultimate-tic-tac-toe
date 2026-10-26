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

test('only canonical app URLs are recognized and matching respects path boundaries', () => {
  assert.equal(stripAppBasePath('/ultimate-tic-tac-toe/api/rooms'), null);
  assert.equal(stripAppBasePath(APP_BASE_PATH), '/');
  assert.equal(stripAppBasePath(`${APP_BASE_PATH}/`), '/');
  assert.equal(stripAppBasePath(`${APP_BASE_PATH}-other/api/rooms`), null);
  assert.equal(appPath(`${APP_BASE_PATH}/`, 'api/rooms'), `${APP_BASE_PATH}/api/rooms`);
});
