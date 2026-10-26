import { DurableObject } from 'cloudflare:workers';
import {
  ROOM_TTL_MS,
  applyRoomMove,
  claimSeat,
  createRoom,
  publicRoom,
  seatForToken,
  type PublicRoom,
  type StoredRoom,
} from '../multiplayer/core.ts';
import type { Player } from '../game.ts';

interface Env {}
interface SocketAttachment { seat: Player | null }
type IncomingMessage = { type?: unknown; token?: unknown; board?: unknown; cell?: unknown; version?: unknown };

const JSON_HEADERS = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' };
const ROOM_KEY = 'room';

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: JSON_HEADERS });
}

function parseMessage(message: string | ArrayBuffer): IncomingMessage | null {
  if (typeof message !== 'string') return null;
  try {
    const value: unknown = JSON.parse(message);
    return value && typeof value === 'object' ? value as IncomingMessage : null;
  } catch {
    return null;
  }
}

function safeSend(socket: WebSocket, message: unknown): void {
  try {
    socket.send(JSON.stringify(message));
  } catch {
    socket.close(1011, 'Unable to send room state');
  }
}

export class GameRoom extends DurableObject<Env> {
  async fetch(request: Request): Promise<Response> {
    const path = new URL(request.url).pathname;
    if (path.endsWith('/create') && request.method === 'POST') return this.create(request);
    if (path.endsWith('/join') && request.method === 'POST') return this.join(request);
    if (path.endsWith('/socket') && request.method === 'GET') return this.openSocket(request);
    return json({ error: 'Not found.' }, 404);
  }

  async webSocketMessage(socket: WebSocket, message: string | ArrayBuffer): Promise<void> {
    const input = parseMessage(message);
    if (!input) {
      socket.close(1003, 'Send a JSON message');
      return;
    }

    const attachment = socket.deserializeAttachment() as SocketAttachment | null;
    if (!attachment?.seat) {
      await this.authenticate(socket, input);
      return;
    }
    if (input.type !== 'move' || !Number.isInteger(input.board) || !Number.isInteger(input.cell)
      || !Number.isInteger(input.version)) {
      safeSend(socket, { type: 'error', message: 'Invalid move.' });
      return;
    }
    await this.move(socket, attachment.seat, input.board as number, input.cell as number, input.version as number);
  }

  async webSocketClose(socket: WebSocket): Promise<void> {
    try { socket.close(); } catch { /* The peer may already have disconnected. */ }
  }

  async alarm(): Promise<void> {
    const room = await this.ctx.storage.get<StoredRoom>(ROOM_KEY);
    if (!room) return;
    const expiresAt = room.updatedAt + ROOM_TTL_MS;
    if (expiresAt > Date.now()) {
      await this.ctx.storage.setAlarm(expiresAt);
      return;
    }
    for (const socket of this.ctx.getWebSockets('room')) {
      try { socket.close(1000, 'Room expired'); } catch { /* Continue closing the remaining sockets. */ }
    }
    await this.ctx.storage.deleteAll();
  }

  private async create(request: Request): Promise<Response> {
    let body: { code?: unknown; hostToken?: unknown };
    try { body = await request.json() as typeof body; }
    catch { return json({ error: 'Invalid request.' }, 400); }
    if (typeof body.code !== 'string' || typeof body.hostToken !== 'string') return json({ error: 'Invalid request.' }, 400);
    const created = await this.ctx.storage.transaction(async (transaction) => {
      if (await transaction.get<StoredRoom>(ROOM_KEY)) return null;
      const room = createRoom(body.code as string, body.hostToken as string, Date.now());
      await transaction.put(ROOM_KEY, room);
      await transaction.setAlarm(room.updatedAt + ROOM_TTL_MS);
      return room;
    });
    if (!created) return json({ error: 'That room code is already in use.' }, 409);
    return json({ room: publicRoom(created) }, 201);
  }

  private async join(request: Request): Promise<Response> {
    let body: { token?: unknown };
    try { body = await request.json() as typeof body; }
    catch { return json({ error: 'Invalid request.' }, 400); }
    if (body.token !== undefined && typeof body.token !== 'string') return json({ error: 'Invalid request.' }, 400);
    const token = typeof body.token === 'string' ? body.token : null;
    const claimed = await this.ctx.storage.transaction(async (transaction) => {
      const current = await transaction.get<StoredRoom>(ROOM_KEY);
      if (!current) return null;
      if (current.updatedAt + ROOM_TTL_MS <= Date.now()) {
        await transaction.delete(ROOM_KEY);
        return null;
      }
      const result = claimSeat(current, token, crypto.randomUUID(), Date.now());
      if (!result.ok) return result;
      await transaction.put(ROOM_KEY, result.room);
      await transaction.setAlarm(result.room.updatedAt + ROOM_TTL_MS);
      return result;
    });
    if (!claimed) return json({ error: 'Room not found or expired.' }, 404);
    if (!claimed.ok) return json({ error: claimed.reason === 'full' ? 'That room already has two players.' : 'This seat token is invalid.' }, 409);
    const room = publicRoom(claimed.room);
    this.broadcast({ type: 'snapshot', room });
    return json({ code: room.code, seat: claimed.seat, token: claimed.token, room });
  }

  private async openSocket(request: Request): Promise<Response> {
    if (request.headers.get('Upgrade')?.toLowerCase() !== 'websocket') return json({ error: 'WebSocket upgrade required.' }, 426);
    const room = await this.ctx.storage.get<StoredRoom>(ROOM_KEY);
    if (!room || room.updatedAt + ROOM_TTL_MS <= Date.now()) return json({ error: 'Room not found or expired.' }, 404);
    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.serializeAttachment({ seat: null } satisfies SocketAttachment);
    this.ctx.acceptWebSocket(server, ['room']);
    return new Response(null, { status: 101, webSocket: client });
  }

  private async authenticate(socket: WebSocket, input: IncomingMessage): Promise<void> {
    if (input.type !== 'hello' || typeof input.token !== 'string') {
      socket.close(1008, 'Authenticate before sending moves');
      return;
    }
    const result = await this.ctx.storage.transaction(async (transaction) => {
      const room = await transaction.get<StoredRoom>(ROOM_KEY);
      if (!room || room.updatedAt + ROOM_TTL_MS <= Date.now()) return null;
      const seat = seatForToken(room, input.token as string);
      if (!seat) return null;
      const updated = { ...room, updatedAt: Date.now() };
      await transaction.put(ROOM_KEY, updated);
      await transaction.setAlarm(updated.updatedAt + ROOM_TTL_MS);
      return { seat, room: updated };
    });
    if (!result) {
      socket.close(1008, 'Invalid room seat');
      return;
    }
    socket.serializeAttachment({ seat: result.seat } satisfies SocketAttachment);
    for (const current of this.ctx.getWebSockets('room')) {
      if (current === socket) continue;
      const attachment = current.deserializeAttachment() as SocketAttachment | null;
      if (attachment?.seat === result.seat) {
        try { current.close(4001, 'Seat reconnected elsewhere'); } catch { /* Continue authentication. */ }
      }
    }
    safeSend(socket, { type: 'snapshot', seat: result.seat, room: publicRoom(result.room) });
  }

  private async move(socket: WebSocket, seat: Player, board: number, cell: number, version: number): Promise<void> {
    const result = await this.ctx.storage.transaction(async (transaction) => {
      const current = await transaction.get<StoredRoom>(ROOM_KEY);
      if (!current) return { ok: false as const, reason: 'expired', room: null };
      const outcome = applyRoomMove(current, seat, board, cell, version, Date.now());
      if (!outcome.ok) return { ok: false as const, reason: outcome.reason, room: current };
      await transaction.put(ROOM_KEY, outcome.room);
      await transaction.setAlarm(outcome.room.updatedAt + ROOM_TTL_MS);
      return { ok: true as const, room: outcome.room };
    });
    if (!result.ok) {
      safeSend(socket, {
        type: 'error',
        message: result.reason === 'not-your-turn' ? 'Wait for your turn.'
          : result.reason === 'not-ready' ? 'Waiting for the other player.'
            : result.reason === 'stale' ? 'The game changed. Board refreshed.'
              : result.reason === 'finished' ? 'This game is already complete.'
                : result.reason === 'illegal' ? 'That move is not legal.' : 'This room has expired.',
        ...(result.room ? { room: publicRoom(result.room) } : {}),
      });
      return;
    }
    this.broadcast({ type: 'snapshot', room: publicRoom(result.room) });
  }

  private broadcast(message: { type: 'snapshot'; room: PublicRoom }): void {
    for (const socket of this.ctx.getWebSockets('room')) {
      const attachment = socket.deserializeAttachment() as SocketAttachment | null;
      if (!attachment?.seat) continue;
      safeSend(socket, message);
    }
  }
}
