import type { GameState } from './game';

export type Difficulty = 'easy' | 'medium' | 'hard';
export type AnalysisMode = 'hint' | 'analysis' | 'review';
export type ComputerMove = { move: number | null; source: 'model' | 'search' };
export type PositionAnalysis = {
  value: number;
  lines: number[][];
  source: 'model' | 'search';
  solved: boolean;
};

type WorkerReply = {
  id: number;
  move?: number | null;
  value?: number;
  lines?: number[][];
  source?: 'model' | 'search';
  solved?: boolean;
  error?: string;
};

type WorkerRequest = {
  action: 'move' | 'analyze';
  state: GameState;
  difficulty?: Difficulty;
  analysisMode?: AnalysisMode;
};

class AIWorkerClient {
  private worker: Worker | null = null;
  private sequence = 0;
  private pending: { reject: (reason: Error) => void; timer: ReturnType<typeof setTimeout> } | null = null;

  request(request: WorkerRequest): Promise<WorkerReply> {
    if (this.pending) this.cancel();
    if (!this.worker) this.worker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
    const worker = this.worker;
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.worker !== worker) return;
        worker.terminate();
        this.worker = null;
        this.pending = null;
        reject(new Error('The analysis took too long. Please try again.'));
      }, 20000);
      this.pending = { reject, timer };
      worker.onmessage = (event: MessageEvent<WorkerReply>) => {
        if (this.worker !== worker) return;
        if (event.data.id !== id || !this.pending) return;
        clearTimeout(timer);
        this.pending = null;
        if (event.data.error) reject(new Error(event.data.error));
        else resolve(event.data);
      };
      worker.onerror = () => {
        if (this.worker !== worker) return;
        clearTimeout(timer);
        this.pending = null;
        worker.terminate();
        this.worker = null;
        reject(new Error('The analysis worker could not start. Please try again.'));
      };
      worker.postMessage({
        ...request,
        id,
        modelUrl: new URL(`${import.meta.env.BASE_URL}models/policy-value.onnx`, document.baseURI).href,
      });
    });
  }

  cancel(): void {
    this.worker?.terminate();
    this.worker = null;
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(new DOMException('Analysis cancelled.', 'AbortError'));
      this.pending = null;
    }
  }

  dispose(): void { this.cancel(); }
}

export class ComputerPlayer {
  private client = new AIWorkerClient();

  async chooseMove(state: GameState, difficulty: Difficulty): Promise<ComputerMove> {
    const reply = await this.client.request({ action: 'move', state, difficulty });
    if (reply.source !== 'model' && reply.source !== 'search') throw new Error('The computer returned an invalid move.');
    return { move: reply.move ?? null, source: reply.source };
  }

  cancel(): void { this.client.cancel(); }
  dispose(): void { this.client.dispose(); }
}

export class PositionAnalyzer {
  private client = new AIWorkerClient();

  async analyze(state: GameState, analysisMode: AnalysisMode = 'analysis'): Promise<PositionAnalysis> {
    const reply = await this.client.request({ action: 'analyze', state, analysisMode });
    if (!Number.isFinite(reply.value) || !Array.isArray(reply.lines)
      || (reply.source !== 'model' && reply.source !== 'search')) {
      throw new Error('The analysis returned an invalid position.');
    }
    return {
      value: Math.max(-1, Math.min(1, reply.value!)),
      lines: reply.lines,
      source: reply.source,
      solved: reply.solved === true,
    };
  }

  cancel(): void { this.client.cancel(); }
  dispose(): void { this.client.dispose(); }
}
