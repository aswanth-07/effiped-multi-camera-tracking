/**
 * Draws a specific frame of a demo clip into a canvas, with an optional
 * highlight box.
 *
 * The hosted build ships the four source clips but no per-appearance scene
 * images — those would have been ~200 KB each. Instead we seek a shared,
 * offscreen <video> to the appearance's timestamp and redraw it on demand,
 * which keeps the deployed bundle small and the frames pixel-exact.
 */

type Box = [number, number, number, number];

const videos = new Map<string, HTMLVideoElement>();
// One seek at a time per clip: concurrent seeks on a single element race.
const queues = new Map<string, Promise<unknown>>();

function element(src: string): HTMLVideoElement {
  let video = videos.get(src);
  if (!video) {
    video = document.createElement("video");
    video.src = src;
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.crossOrigin = "anonymous";
    videos.set(src, video);
  }
  return video;
}

function once(target: HTMLVideoElement, event: string, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error(`timeout waiting for ${event}`));
    }, timeoutMs);
    function done() {
      cleanup();
      resolve();
    }
    function failed() {
      cleanup();
      reject(new Error(`error waiting for ${event}`));
    }
    function cleanup() {
      window.clearTimeout(timer);
      target.removeEventListener(event, done);
      target.removeEventListener("error", failed);
    }
    target.addEventListener(event, done, { once: true });
    target.addEventListener("error", failed, { once: true });
  });
}

async function ready(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= 1) return;
  video.load();
  await once(video, "loadedmetadata", 15000);
}

async function seek(video: HTMLVideoElement, timeS: number): Promise<void> {
  const duration = Number.isFinite(video.duration) ? video.duration : timeS;
  const target = Math.max(0, Math.min(timeS, Math.max(0, duration - 0.001)));
  if (Math.abs(video.currentTime - target) < 1e-3 && video.readyState >= 2) return;
  const seeked = once(video, "seeked", 15000);
  video.currentTime = target;
  await seeked;
  if (video.readyState < 2) await once(video, "loadeddata", 15000);
}

/** Serialise work per clip so seeks don't interleave. */
function enqueue<T>(src: string, task: () => Promise<T>): Promise<T> {
  const previous = queues.get(src) ?? Promise.resolve();
  const next = previous.then(task, task);
  queues.set(
    src,
    next.catch(() => undefined)
  );
  return next;
}

export type DrawFrameOptions = {
  box?: Box;
  boxColor?: string;
  boxLabel?: string;
  /** Render at this width; height follows the clip's aspect ratio. */
  width?: number;
  dim?: boolean;
};

export async function drawFrame(
  canvas: HTMLCanvasElement,
  src: string,
  timeS: number,
  options: DrawFrameOptions = {}
): Promise<void> {
  await enqueue(src, async () => {
    const video = element(src);
    await ready(video);
    await seek(video, timeS);

    const naturalWidth = video.videoWidth || 960;
    const naturalHeight = video.videoHeight || 540;
    const width = options.width ?? naturalWidth;
    const scale = width / naturalWidth;
    const height = Math.round(naturalHeight * scale);

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, width, height);

    if (options.dim) {
      ctx.fillStyle = "rgba(7, 10, 15, 0.45)";
      ctx.fillRect(0, 0, width, height);
    }

    const box = options.box;
    if (box) {
      const [x1, y1, x2, y2] = box.map((value) => value * scale) as Box;
      const boxWidth = Math.max(1, x2 - x1);
      const boxHeight = Math.max(1, y2 - y1);

      if (options.dim) {
        // Punch the subject back out of the dimmed plate.
        ctx.save();
        ctx.beginPath();
        ctx.rect(x1, y1, boxWidth, boxHeight);
        ctx.clip();
        ctx.drawImage(video, 0, 0, width, height);
        ctx.restore();
      }

      const color = options.boxColor ?? "#22c7b8";
      ctx.lineWidth = Math.max(2, Math.round(2 * scale));
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      if (options.boxLabel) {
        ctx.font = `600 ${Math.max(11, Math.round(13 * scale))}px Inter, system-ui, sans-serif`;
        const padding = 5;
        const textWidth = ctx.measureText(options.boxLabel).width;
        const labelHeight = Math.max(16, Math.round(18 * scale));
        const labelY = y1 - labelHeight < 0 ? y1 : y1 - labelHeight;
        ctx.fillStyle = color;
        ctx.fillRect(x1, labelY, textWidth + padding * 2, labelHeight);
        ctx.fillStyle = "#04120f";
        ctx.fillText(options.boxLabel, x1 + padding, labelY + labelHeight - padding);
      }
    }
  });
}

export function releaseFrameCache(): void {
  videos.forEach((video) => {
    video.removeAttribute("src");
    video.load();
  });
  videos.clear();
  queues.clear();
}
