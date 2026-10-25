import { env, InferenceSession, Tensor } from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import runtimeUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import { encodeState, INPUT_SIZE, type Evaluator } from './ai';

// One CPU thread works without cross-origin isolation or GPU support. All
// runtime assets are emitted by Vite and served from this site's own origin.
env.wasm.numThreads = 1;
env.wasm.proxy = false;
env.wasm.wasmPaths = {
  wasm: new URL(wasmUrl, import.meta.url).href,
  mjs: new URL(runtimeUrl, import.meta.url).href,
};

export type ModelEvaluator = Evaluator & { dispose(): Promise<void> };

export async function createEvaluator(modelUrl: string): Promise<ModelEvaluator> {
  const response = await fetch(modelUrl, { signal: AbortSignal.timeout(8000) });
  if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
  const bytes = await response.arrayBuffer();
  const session = await InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  if (session.inputNames.join() !== 'state' || session.outputNames.join() !== 'policy,value') {
    await session.release();
    throw new Error('Unsupported policy/value model interface.');
  }
  const evaluate: Evaluator = async (state) => {
    const input = new Tensor('float32', encodeState(state), [1, INPUT_SIZE]);
    let output: InferenceSession.ReturnType | undefined;
    try {
      output = await session.run({ state: input });
      const policy = Float32Array.from(output.policy.data as Float32Array);
      const value = Number(output.value.data[0]);
      if (output.policy.dims.join() !== '1,81' || output.value.dims.join() !== '1,1'
        || policy.length !== 81 || !policy.every(Number.isFinite) || !Number.isFinite(value) || Math.abs(value) > 1.001) {
        throw new Error('Invalid policy/value model output.');
      }
      return { policy, value };
    } finally {
      input.dispose();
      if (output) for (const tensor of Object.values(output)) tensor.dispose();
    }
  };
  return Object.assign(evaluate, { dispose: () => session.release() });
}
