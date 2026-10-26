import { createGame, isLegal, play, type GameState, type Player } from '../game.ts';

export const ROOM_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
export const ROOM_CODE_LENGTH = 8;
export const ROOM_TTL_MS = 24 * 60 * 60 * 1000;

export type RoomStatus = 'waiting' | 'playing' | 'complete';

export interface StoredRoom {
  code: string;
  state: GameState;
  version: number;
  moves: number[];
  hostToken: string;
  guestToken: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface PublicRoom {
  code: string;
  state: GameState;
  version: number;
  moves: number[];
  players: number;
  status: RoomStatus;
}

export type ClaimResult =
  | { ok: true; room: StoredRoom; seat: Player; token: string }
  | { ok: false; reason: 'unauthorized' | 'full' };

export type MoveResult =
  | { ok: true; room: StoredRoom }
  | { ok: false; reason: 'not-ready' | 'finished' | 'stale' | 'not-your-turn' | 'illegal' };

export function normalizeRoomCode(value: string): string {
  return value.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

export function isRoomCode(value: string): boolean {
  return value.length === ROOM_CODE_LENGTH
    && [...value].every((character) => ROOM_CODE_ALPHABET.includes(character));
}

export function createRoom(code: string, hostToken: string, now: number): StoredRoom {
  if (!isRoomCode(code) || !hostToken) throw new Error('Invalid room credentials.');
  return {
    code,
    state: createGame(),
    version: 0,
    moves: [],
    hostToken,
    guestToken: null,
    createdAt: now,
    updatedAt: now,
  };
}

export function claimSeat(room: StoredRoom, token: string | null, newGuestToken: string, now: number): ClaimResult {
  if (token && token === room.hostToken) {
    return { ok: true, room, seat: 'X', token: room.hostToken };
  }
  if (token && token === room.guestToken) {
    return { ok: true, room, seat: 'O', token: room.guestToken! };
  }
  if (token) return { ok: false, reason: 'unauthorized' };
  if (room.guestToken) return { ok: false, reason: 'full' };
  if (!newGuestToken) throw new Error('A guest token is required.');
  const updated = { ...room, guestToken: newGuestToken, updatedAt: now };
  return { ok: true, room: updated, seat: 'O', token: newGuestToken };
}

export function seatForToken(room: StoredRoom, token: string): Player | null {
  if (token && token === room.hostToken) return 'X';
  if (token && token === room.guestToken) return 'O';
  return null;
}

export function applyRoomMove(
  room: StoredRoom,
  seat: Player,
  board: number,
  cell: number,
  expectedVersion: number,
  now: number,
): MoveResult {
  if (!room.guestToken) return { ok: false, reason: 'not-ready' };
  if (room.state.winner) return { ok: false, reason: 'finished' };
  if (!Number.isInteger(expectedVersion) || expectedVersion !== room.version) return { ok: false, reason: 'stale' };
  if (seat !== room.state.turn) return { ok: false, reason: 'not-your-turn' };
  if (!isLegal(room.state, board, cell)) return { ok: false, reason: 'illegal' };
  const state = play(room.state, board, cell);
  return {
    ok: true,
    room: {
      ...room,
      state,
      version: room.version + 1,
      moves: [...room.moves, board * 9 + cell],
      updatedAt: now,
    },
  };
}

export function publicRoom(room: StoredRoom): PublicRoom {
  return {
    code: room.code,
    state: room.state,
    version: room.version,
    moves: room.moves,
    players: room.guestToken ? 2 : 1,
    status: room.state.winner ? 'complete' : room.guestToken ? 'playing' : 'waiting',
  };
}
