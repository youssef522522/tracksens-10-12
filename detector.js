// detector.js — run a YOLOv8 ONNX model in the browser with onnxruntime-web.
// No bundler needed: ort is loaded from a pinned CDN build.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";

// !! Order MUST match the `names:` order in your training data.yaml !!
export const CLASSES = ["fastener", "sleeper", "bolt"];

const INPUT = 640; // the imgsz you exported with

let session = null;
let inputName = null;

// Reusable canvas for preprocessing
const canvas = document.createElement("canvas");
canvas.width = canvas.height = INPUT;
const ctx = canvas.getContext("2d", { willReadFrequently: true });

export async function loadModel(url = "./best.onnx") {
  ort.env.wasm.numThreads = 1; // avoids needing COOP/COEP headers on Cloudflare
  session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
  });
  inputName = session.inputNames[0];
  return session;
}

// Letterbox the source (img / video / canvas) into 640x640, return CHW float tensor.
function preprocess(source) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.min(INPUT / sw, INPUT / sh);
  const nw = Math.round(sw * scale);
  const nh = Math.round(sh * scale);
  const padX = (INPUT - nw) / 2;
  const padY = (INPUT - nh) / 2;

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, INPUT, INPUT);
  ctx.drawImage(source, padX, padY, nw, nh);

  const { data } = ctx.getImageData(0, 0, INPUT, INPUT);
  const area = INPUT * INPUT;
  const f = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    f[i] = data[i * 4] / 255;
    f[i + area] = data[i * 4 + 1] / 255;
    f[i + 2 * area] = data[i * 4 + 2] / 255;
  }
  return {
    tensor: new ort.Tensor("float32", f, [1, 3, INPUT, INPUT]),
    scale, padX, padY, sw, sh,
  };
}

function iouOf(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

// Per-class non-max suppression
function nms(boxes, thr) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => k.classId !== b.classId || iouOf(k, b) < thr)) {
      keep.push(b);
    }
  }
  return keep;
}

// YOLOv8 output layout: [1, 4 + numClasses, numAnchors] (cx, cy, w, h, scores...)
function postprocess(output, meta, confThr, iouThr) {
  const data = output.data;
  const rows = output.dims[1];
  const n = output.dims[2];
  const nc = rows - 4;
  const boxes = [];

  for (let i = 0; i < n; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * n + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < confThr) continue;

    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
    // Undo letterbox -> original image coordinates
    const x1 = (cx - w / 2 - meta.padX) / meta.scale;
    const y1 = (cy - h / 2 - meta.padY) / meta.scale;
    const x2 = (cx + w / 2 - meta.padX) / meta.scale;
    const y2 = (cy + h / 2 - meta.padY) / meta.scale;

    boxes.push({
      x1: Math.max(0, x1), y1: Math.max(0, y1),
      x2: Math.min(meta.sw, x2), y2: Math.min(meta.sh, y2),
      score: best, classId: cls, label: CLASSES[cls] ?? `class${cls}`,
    });
  }
  return nms(boxes, iouThr);
}

// source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
// returns [{x1,y1,x2,y2,score,classId,label}] in the source's pixel coordinates
export async function detect(source, { conf = 0.25, iou = 0.45 } = {}) {
  if (!session) throw new Error("Call loadModel() first");
  const meta = preprocess(source);
  const out = await session.run({ [inputName]: meta.tensor });
  return postprocess(out[session.outputNames[0]], meta, conf, iou);
}

const COLORS = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b"];

// Draw boxes on a 2D canvas context whose size matches the source image/video.
export function drawDetections(c2d, dets) {
  c2d.lineWidth = Math.max(2, c2d.canvas.width / 400);
  c2d.font = `${Math.max(12, c2d.canvas.width / 50)}px sans-serif`;
  for (const d of dets) {
    const color = COLORS[d.classId % COLORS.length];
    c2d.strokeStyle = color;
    c2d.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const text = `${d.label} ${(d.score * 100).toFixed(0)}%`;
    const tw = c2d.measureText(text).width + 6;
    const th = parseInt(c2d.font, 10) + 4;
    c2d.fillStyle = color;
    c2d.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
    c2d.fillStyle = "#fff";
    c2d.fillText(text, d.x1 + 3, Math.max(th - 4, d.y1 - 4));
  }
}
