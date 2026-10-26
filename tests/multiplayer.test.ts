import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ROOM_CODE_ALPHABET,
  applyRoomMove,
  claimSeat,
  createRoom,
  isRoomCode,
  normalizeRoomCode,
  publicRoom,
  seatForToken,
} from '../src/multiplayer/core.ts';

const CODE = 'ABCD2345';

test('room codes normalize for sharing and reject ambiguous or malformed values', () => {
  assert.equal(ROOM_CODE_ALPHABET.length, 32);
  assert.equal(normalizeRoomCode(' abcd-2345 '), CODE);
  assert.equal(isRoomCode(CODE), true);
  assert.equal(isRoomCode('ABCD0123'), false);
  assert.equal(isRoomCode('short'), false);
});

test('room claims assign X to the host, O to one guest, and restore only valid seat tokens', () => {
  const room = createRoom(CODE, 'host-secret', 10);
  const host = claimSeat(room, 'host-secret', 'unused', 11);
  assert.equal(host.ok, true);
  if (!host.ok) return;
  assert.equal(host.seat, 'X');
  assert.equal(host.room.guestToken, null);

  const guest = claimSeat(host.room, null, 'guest-secret', 12);
  assert.equal(guest.ok, true);
  if (!guest.ok) return;
  assert.equal(guest.seat, 'O');
  assert.equal(guest.room.guestToken, 'guest-secret');
  assert.equal(claimSeat(guest.room, null, 'third-secret', 13).ok, false);
  assert.equal(seatForToken(guest.room, 'host-secret'), 'X');
  assert.equal(seatForToken(guest.room, 'guest-secret'), 'O');
  assert.equal(seatForToken(guest.room, 'wrong-secret'), null);
  assert.equal(claimSeat(guest.room, 'wrong-secret', 'unused', 14).ok, false);
});

test('room moves are turn, version, legality, and readiness checked before state changes', () => {
  const waiting = createRoom(CODE, 'host-secret', 0);
  assert.deepEqual(applyRoomMove(waiting, 'X', 0, 0, 0, 1), { ok: false, reason: 'not-ready' });
  const claim = claimSeat(waiting, null, 'guest-secret', 2);
  assert.equal(claim.ok, true);
  if (!claim.ok) return;

  assert.deepEqual(applyRoomMove(claim.room, 'O', 0, 0, 0, 3), { ok: false, reason: 'not-your-turn' });
  const first = applyRoomMove(claim.room, 'X', 0, 0, 0, 4);
  assert.equal(first.ok, true);
  if (!first.ok) return;
  assert.equal(first.room.state.boards[0][0], 'X');
  assert.equal(first.room.version, 1);
  assert.deepEqual(first.room.moves, [0]);

  assert.deepEqual(applyRoomMove(first.room, 'O', 0, 1, 0, 5), { ok: false, reason: 'stale' });
  assert.deepEqual(applyRoomMove(first.room, 'O', 1, 1, 1, 6), { ok: false, reason: 'illegal' });
  const second = applyRoomMove(first.room, 'O', 0, 1, 1, 7);
  assert.equal(second.ok, true);
  if (second.ok) {
    assert.equal(second.room.state.boards[0][1], 'O');
    assert.deepEqual(second.room.moves, [0, 1]);
  }
});

test('public room snapshots never expose seat tokens', () => {
  const created = createRoom(CODE, 'host-secret', 10);
  const guest = claimSeat(created, null, 'guest-secret', 11);
  assert.equal(guest.ok, true);
  if (!guest.ok) return;
  const snapshot = publicRoom(guest.room);
  assert.equal(snapshot.players, 2);
  assert.equal(snapshot.status, 'playing');
  assert.equal('hostToken' in snapshot, false);
  assert.equal('guestToken' in snapshot, false);
  assert.equal(JSON.stringify(snapshot).includes('secret'), false);
});
