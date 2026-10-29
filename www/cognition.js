import { env, pipeline } from './transformers.min.js';

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = './models/';
env.useBrowserCache = false;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.wasmPaths = './ort/';

let _extractor = null;
let _err = null;
const _ready = (async () => {
  try {
    _extractor = await pipeline('feature-extraction', 'Xenova/bge-small-zh-v1.5', { quantized: true });
    console.log('[cog] model ready');
  } catch (e) {
    _err = e;
    console.error('[cog] init failed', e);
  }
})();

const _cache = new Map();
async function embedOne(text) {
  await _ready;
  if (!_extractor) throw _err || new Error('cog unavailable');
  const key = String(text || '').slice(0, 512);
  const hit = _cache.get(key);
  if (hit) return hit;
  const out = await _extractor(key, { pooling: 'mean', normalize: true });
  const v = out.data;
  if (_cache.size >= 500) _cache.delete(_cache.keys().next().value);
  _cache.set(key, v);
  return v;
}
function cos(a, b) {
  let s = 0; const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

window.cog = {
  ready: () => _ready.then(() => !!_extractor),
  isReady: () => !!_extractor,
  error: () => _err,
  embed: embedOne,
  sim: async (a, b) => cos(await embedOne(a), await embedOne(b)),
  simHist: async (msg, hist) => {
    const vs = await embedOne(msg);
    let best = -1, bestIdx = -1;
    for (let i = 0; i < hist.length; i++) {
      const s = cos(vs, await embedOne(hist[i]));
      if (s > best) { best = s; bestIdx = i; }
    }
    return { best, bestIdx };
  }
};
console.log('[cog] module loaded');
