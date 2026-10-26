import { isRoomCode, normalizeRoomCode, type PublicRoom } from './multiplayer/core';
import type { Player } from './game';

const ROOM_TOKEN_PREFIX = 'ultimate-tic-tac-toe:room-token:';

export interface OnlineSeat {
  code: string;
  seat: Player;
  token: string;
  room: PublicRoom;
}

export type OnlineConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'closed' | 'error';

export type OnlineSocketMessage =
  | { type: 'snapshot'; room: PublicRoom; seat?: Player }
  | { type: 'error'; message: string; room?: PublicRoom };

export interface OnlineRoomCallbacks {
  onStatus(status: OnlineConnectionStatus): void;
  onMessage(message: OnlineSocketMessage): void;
  onError(message: string): void;
}

function readToken(code: string): string | null {
  try { return localStorage.getItem(ROOM_TOKEN_PREFIX + code); }
  catch { return null; }
}

function saveToken(code: string, token: string): void {
  try { localStorage.setItem(ROOM_TOKEN_PREFIX + code, token); }
  catch { /* The current tab can still play if browser storage is unavailable. */ }
}

async function requestJson<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(url, { ...init, headers: { 'Content-Type': 'application/json', ...init.headers } }); }
  catch { throw new Error('Could not reach the room server. Check the connection and try again.'); }
  let payload: { error?: unknown } & Record<string, unknown>;
  try { payload = await response.json() as typeof payload; }
  catch { throw new Error('The room server returned an unreadable response.'); }
  if (!response.ok) {
    throw new Error(typeof payload.error === 'string' ? payload.error : 'The room request failed.');
  }
  return payload as T;
}

export async function createOnlineRoom(): Promise<OnlineSeat> {
  const result = await requestJson<OnlineSeat>('/api/rooms', { method: 'POST', body: '{}' });
  if (!isRoomCode(result.code) || result.seat !== 'X' || typeof result.token !== 'string') {
    throw new Error('The room server returned invalid room details.');
  }
  saveToken(result.code, result.token);
  return result;
}

export async function joinOnlineRoom(input: string): Promise<OnlineSeat> {
  const code = normalizeRoomCode(input);
  if (!isRoomCode(code)) throw new Error('Enter the 8-character room code.');
  const token = readToken(code);
  const result = await requestJson<OnlineSeat>(`/api/rooms/${code}/join`, {
    method: 'POST',
    body: JSON.stringify(token ? { token } : {}),
  });
  if (result.code !== code || (result.seat !== 'X' && result.seat !== 'O') || typeof result.token !== 'string') {
    throw new Error('The room server returned invalid seat details.');
  }
  saveToken(code, result.token);
  return result;
}

export function roomInviteUrl(code: string): string {
  const url = new URL(window.location.href);
  url.searchParams.set('room', code);
  return url.toString();
}

export class OnlineRoomConnection {
  private socket: WebSocket | null = null;
  private closedByUser = false;
  private retryTimer: number | null = null;
  private retryDelay = 500;

  constructor(private readonly seat: OnlineSeat, private readonly callbacks: OnlineRoomCallbacks) {}

  connect(): void {
    this.closedByUser = false;
    this.open(false);
  }

  close(): void {
    this.closedByUser = true;
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.socket?.close(1000, 'Room left');
    this.socket = null;
    this.callbacks.onStatus('closed');
  }

  sendMove(board: number, cell: number, version: number): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify({ type: 'move', board, cell, version }));
    return true;
  }

  private open(reconnecting: boolean): void {
    if (this.closedByUser) return;
    this.callbacks.onStatus(reconnecting ? 'reconnecting' : 'connecting');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/api/rooms/${this.seat.code}/socket`);
    this.socket = socket;
    socket.addEventListener('open', () => {
      socket.send(JSON.stringify({ type: 'hello', token: this.seat.token }));
    });
    socket.addEventListener('message', (event: MessageEvent<string>) => {
      let message: OnlineSocketMessage;
      try { message = JSON.parse(event.data) as OnlineSocketMessage; }
      catch {
        this.callbacks.onError('The room sent an invalid update.');
        return;
      }
      if (message.type === 'snapshot') {
        this.retryDelay = 500;
        this.callbacks.onStatus('connected');
        this.callbacks.onMessage(message);
      } else if (message.type === 'error') {
        this.callbacks.onError(message.message);
        this.callbacks.onMessage(message);
      }
    });
    socket.addEventListener('close', (event) => {
      if (this.closedByUser) return;
      if (event.code === 1008 || event.code === 4001 || event.code === 1000 && event.reason === 'Room expired') {
        this.callbacks.onStatus('error');
        this.callbacks.onError(event.code === 4001
          ? 'This seat is active in another tab.'
          : event.reason === 'Room expired' ? 'This room has expired.' : 'This room seat could not be restored. Start or join a room again.');
        return;
      }
      this.scheduleReconnect();
    });
    socket.addEventListener('error', () => socket.close());
  }

  private scheduleReconnect(): void {
    if (this.closedByUser || this.retryTimer !== null) return;
    this.callbacks.onStatus('reconnecting');
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.open(true);
    }, this.retryDelay);
    this.retryDelay = Math.min(10_000, this.retryDelay * 2);
  }
}
