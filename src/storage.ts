export const STORAGE_KEYS = {
  game: 'ultimate-tic-tac-toe:v1',
  opponent: 'ultimate-tic-tac-toe:opponent',
  theme: 'ultimate-tic-tac-toe:theme',
} as const;

export const LEGACY_STORAGE_KEYS = {
  game: 'meta-tic-tac-toe:v1',
  opponent: 'meta-tic-tac-toe:opponent',
  theme: 'meta-tic-tac-toe:theme',
} as const;

type StorageProvider = () => Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

const storagePairs = [
  [STORAGE_KEYS.game, LEGACY_STORAGE_KEYS.game],
  [STORAGE_KEYS.opponent, LEGACY_STORAGE_KEYS.opponent],
  [STORAGE_KEYS.theme, LEGACY_STORAGE_KEYS.theme],
] as const;

export function migrateLegacyStorage(getStorage: StorageProvider = () => localStorage): void {
  try {
    const storage = getStorage();
    for (const [currentKey, legacyKey] of storagePairs) {
      try {
        if (storage.getItem(currentKey) !== null) continue;
        const legacyValue = storage.getItem(legacyKey);
        if (legacyValue === null) continue;
        storage.setItem(currentKey, legacyValue);
        if (storage.getItem(currentKey) === legacyValue) storage.removeItem(legacyKey);
      } catch {
        // Keep the old value if storage is unavailable or the copy fails.
      }
    }
  } catch {
    // The game remains usable if this origin cannot access browser storage.
  }
}

export function readMigratedStorage(
  currentKey: string,
  legacyKey: string,
  getStorage: StorageProvider = () => localStorage,
): string | null {
  try {
    const storage = getStorage();
    return storage.getItem(currentKey) ?? storage.getItem(legacyKey);
  } catch {
    return null;
  }
}
