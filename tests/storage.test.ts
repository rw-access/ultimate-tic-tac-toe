import assert from 'node:assert/strict';
import test from 'node:test';
import { LEGACY_STORAGE_KEYS, migrateLegacyStorage, readMigratedStorage, STORAGE_KEYS } from '../src/storage.ts';

function memoryStorage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    values,
    getItem(key: string) { return values.get(key) ?? null; },
    setItem(key: string, value: string) { values.set(key, value); },
    removeItem(key: string) { values.delete(key); },
  };
}

test('migrates game and preferences to the Ultimate Tic-Tac-Toe storage namespace', () => {
  const oldGame = '{"version":1,"state":{"moves":3}}';
  const oldOpponent = '{"mode":"computer","difficulty":"hard"}';
  const storage = memoryStorage({
    [LEGACY_STORAGE_KEYS.game]: oldGame,
    [LEGACY_STORAGE_KEYS.opponent]: oldOpponent,
    [LEGACY_STORAGE_KEYS.theme]: 'dark',
  });

  migrateLegacyStorage(() => storage);

  assert.equal(storage.getItem(STORAGE_KEYS.game), oldGame);
  assert.equal(storage.getItem(STORAGE_KEYS.opponent), oldOpponent);
  assert.equal(storage.getItem(STORAGE_KEYS.theme), 'dark');
  assert.equal(storage.getItem(LEGACY_STORAGE_KEYS.game), null);
  assert.equal(storage.getItem(LEGACY_STORAGE_KEYS.opponent), null);
  assert.equal(storage.getItem(LEGACY_STORAGE_KEYS.theme), null);
});

test('new storage wins and an inaccessible destination leaves the old value readable', () => {
  const storage = memoryStorage({
    [STORAGE_KEYS.theme]: 'light',
    [LEGACY_STORAGE_KEYS.theme]: 'dark',
    [LEGACY_STORAGE_KEYS.game]: 'saved-game',
  });
  const readStorage = () => storage;

  migrateLegacyStorage(readStorage);
  assert.equal(readMigratedStorage(STORAGE_KEYS.theme, LEGACY_STORAGE_KEYS.theme, readStorage), 'light');

  const blocked = {
    getItem(key: string) { return key === LEGACY_STORAGE_KEYS.game ? 'saved-game' : null; },
    setItem() { throw new Error('quota exceeded'); },
    removeItem() {},
  };
  migrateLegacyStorage(() => blocked);
  assert.equal(readMigratedStorage(STORAGE_KEYS.game, LEGACY_STORAGE_KEYS.game, () => blocked), 'saved-game');
});
