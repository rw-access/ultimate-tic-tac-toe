// Binary request: little-endian uint32 row count and value-only flag, then
// 280 uint8 features per row. Reply: float32 policy (if requested), then values.
const { once } = require('node:events');
const { endianness } = require('node:os');
const path = require('node:path');

async function run() {
  if (endianness() !== 'LE') throw new Error('The worker requires little-endian float32');
  const [model, moduleDir] = process.argv.slice(2);
  const ort = require(path.join(path.resolve(moduleDir), 'onnxruntime-node'));
  if (!ort.listSupportedBackends().some(backend => backend.name === 'cuda')) {
    throw new Error('onnxruntime-node does not list CUDA support');
  }
  // An explicit CUDA EP request must succeed. Never retry with CPU.
  const session = await ort.InferenceSession.create(model, {
    executionProviders: [{ name: 'cuda', deviceId: 0 }],
    graphOptimizationLevel: 'all',
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });
  if (session.inputNames.join() !== 'state' || !['policy', 'value'].every(name => session.outputNames.includes(name))) {
    throw new Error('Expected a native policy/value ONNX interface');
  }

  const chunks = process.stdin[Symbol.asyncIterator]();
  let remaining = Buffer.alloc(0);
  async function read(length, allowEof = false) {
    const result = Buffer.allocUnsafe(length);
    let offset = 0;
    while (offset < length) {
      if (!remaining.length) {
        const chunk = await chunks.next();
        if (chunk.done && allowEof && offset === 0) return undefined;
        if (chunk.done) throw new Error('Truncated input batch');
        remaining = chunk.value;
      }
      const count = remaining.copy(result, offset, 0, length - offset);
      remaining = remaining.subarray(count);
      offset += count;
    }
    return result;
  }
  async function write(buffer) {
    if (!process.stdout.write(buffer)) await once(process.stdout, 'drain');
  }

  try {
    await write(Buffer.from('UO01'));
    while (true) {
      const header = await read(8, true);
      if (!header) return;
      const count = header.readUInt32LE(0), valueOnly = header.readUInt32LE(4);
      if (count < 1 || count > 65536 || valueOnly > 1) throw new Error('Invalid batch header');
      const raw = await read(count * 280);
      const input = new ort.Tensor('float32', Float32Array.from(raw), [count, 280]);
      const wanted = valueOnly ? ['value'] : ['policy', 'value'];
      let result;
      try {
        result = await session.run({ state: input }, wanted);
        for (const name of wanted) {
          const t = result[name];
          const width = name === 'value' ? 1 : 81;
          if (t.type !== 'float32' || t.dims.join() !== [count, width].join()) {
            throw new Error('Unexpected teacher tensor shape or dtype for ' + name);
          }
          // Copy before releasing the native tensor; stdout may buffer writes.
          const bytes = new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength);
          await write(Buffer.from(bytes));
        }
      } finally {
        input.dispose();
        if (result) for (const tensor of Object.values(result)) tensor.dispose();
      }
    }
  } finally {
    await session.release();
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; process.stdin.destroy(); });
