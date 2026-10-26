import { test, expect } from '@playwright/test';
import { createGame, play } from '../../src/game';
import type { PublicRoom } from '../../src/multiplayer/core';

const CODE = 'ABCD2345';
const APP_PATH = '/play/ultimate-tic-tac-toe';

function room(state = createGame(), players = 2, version = 0, moves: number[] = []): PublicRoom {
  return { code: CODE, state, players, version, moves, status: players === 1 ? 'waiting' : 'playing' };
}

test('online host creates a shareable room and submits authoritative moves', async ({ page }) => {
  let current = room();
  const waitingRoom = room(createGame(), 1);
  await page.route(`**${APP_PATH}/api/rooms`, route => route.fulfill({
    status: 201,
    contentType: 'application/json',
    body: JSON.stringify({ code: CODE, seat: 'X', token: 'host-token', room: waitingRoom }),
  }));
  await page.routeWebSocket(/\/ultimate-tic-tac-toe\/api\/rooms\/ABCD2345\/socket$/, socket => {
    socket.onMessage(message => {
      const input = JSON.parse(String(message)) as { type: string; board?: number; cell?: number; version?: number };
      if (input.type === 'hello') {
        socket.send(JSON.stringify({ type: 'snapshot', seat: 'X', room: current }));
      } else if (input.type === 'move' && input.board !== undefined && input.cell !== undefined) {
        const state = play(current.state, input.board, input.cell);
        current = room(state, 2, current.version + 1, [...current.moves, input.board * 9 + input.cell]);
        socket.send(JSON.stringify({ type: 'snapshot', room: current }));
      }
    });
  });

  await page.goto(`${APP_PATH}/`);
  await page.getByRole('button', { name: 'New game' }).click();
  const menu = page.getByRole('dialog', { name: 'New game' });
  await menu.getByRole('button', { name: 'Play online' }).click();
  await menu.getByRole('button', { name: 'Create room' }).click();
  await expect(page.getByRole('region', { name: 'Online game room' })).toContainText('ABCD 2345');
  await expect(page.getByRole('button', { name: 'Undo' })).toBeDisabled();
  await expect(page.locator('[data-board="0"][data-cell="0"]')).toBeEnabled();
  await page.locator('[data-board="0"][data-cell="0"]').click();
  await expect(page.locator('[data-board="0"][data-cell="0"]')).toContainText('X');
  await expect(page).toHaveURL(new RegExp(`${APP_PATH}/\\?room=ABCD2345`));
});

test('an invite link claims a seat and reconnects from the room URL', async ({ page }) => {
  const initial = room();
  await page.route(`**${APP_PATH}/api/rooms/${CODE}/join`, route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ code: CODE, seat: 'O', token: 'guest-token', room: initial }),
  }));
  await page.routeWebSocket(/\/ultimate-tic-tac-toe\/api\/rooms\/ABCD2345\/socket$/, socket => {
    socket.onMessage(message => {
      const input = JSON.parse(String(message)) as { type: string };
      if (input.type === 'hello') socket.send(JSON.stringify({ type: 'snapshot', seat: 'O', room: initial }));
    });
  });

  await page.goto(`${APP_PATH}/?room=${CODE}`);
  await expect(page.getByRole('region', { name: 'Online game room' })).toContainText('ABCD 2345');
  await expect(page.locator('.turn-title')).toHaveText('X to play');
  await expect(page.locator('[data-board="0"][data-cell="0"]')).toBeDisabled();
  await expect(page).toHaveURL(new RegExp(`${APP_PATH}/\\?room=${CODE}`));
});
