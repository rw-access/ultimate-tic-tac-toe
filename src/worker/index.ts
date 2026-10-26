import { isRoomCode, ROOM_CODE_ALPHABET, ROOM_CODE_LENGTH } from '../multiplayer/core.ts';
import { APP_BASE_PATH, stripAppBasePath } from '../app-path.ts';

interface Env {
  ASSETS: Fetcher;
  ROOMS: DurableObjectNamespace;
}

export { GameRoom } from './room.ts';

const JSON_HEADERS = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' };
const ROOM_ATTEMPTS = 8;

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: JSON_HEADERS });
}

function randomToken(): string {
  return crypto.randomUUID();
}

function randomRoomCode(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(ROOM_CODE_LENGTH));
  return [...bytes].map((byte) => ROOM_CODE_ALPHABET[byte % ROOM_CODE_ALPHABET.length]).join('');
}

async function createRoom(env: Env): Promise<Response> {
  for (let attempt = 0; attempt < ROOM_ATTEMPTS; attempt += 1) {
    const code = randomRoomCode();
    const hostToken = randomToken();
    const room = env.ROOMS.get(env.ROOMS.idFromName(code));
    const response = await room.fetch('https://room.internal/api/rooms/create', {
      method: 'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify({ code, hostToken }),
    });
    if (response.status === 409) continue;
    if (!response.ok) return json({ error: 'Could not create a room.' }, 503);
    const result = await response.json() as { room: unknown };
    return json({ code, seat: 'X', token: hostToken, room: result.room }, 201);
  }
  return json({ error: 'Could not find an available room code. Try again.' }, 503);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/') {
      url.pathname = `${APP_BASE_PATH}/`;
      return Response.redirect(url, 302);
    }
    const pathname = stripAppBasePath(url.pathname);
    if (!pathname?.startsWith('/api/')) return env.ASSETS.fetch(request);
    if (pathname === '/api/health' && request.method === 'GET') return json({ ok: true });
    if (pathname === '/api/rooms' && request.method === 'POST') return createRoom(env);

    const roomMatch = pathname.match(/^\/api\/rooms\/([^/]+)\/(join|socket)$/);
    if (roomMatch && (request.method === 'POST' && roomMatch[2] === 'join'
      || request.method === 'GET' && roomMatch[2] === 'socket')) {
      const code = decodeURIComponent(roomMatch[1]).toUpperCase();
      if (!isRoomCode(code)) return json({ error: 'Enter a valid room code.' }, 400);
      return env.ROOMS.get(env.ROOMS.idFromName(code)).fetch(request);
    }

    return json({ error: 'Not found.' }, 404);
  },
} satisfies ExportedHandler<Env>;
