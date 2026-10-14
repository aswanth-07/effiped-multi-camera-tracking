import { useCallback, useEffect, useRef } from "react";

import { frameTime } from "./frameGrabber";
import { trackBoxesForFrame, type Settings, type TrackBox } from "../engine/pipeline";
import type { WorkbenchVideo } from "../types";

const INDEXED = "134, 255, 107";
const UNINDEXED = "210, 153, 34";

/**
 * Where the picture sits inside the element once object-fit has placed it, and
 * how many CSS pixels one source pixel is. Scaling by width alone is only right
 * while the element has exactly the clip's aspect ratio.
 */
function contentBox(element: HTMLVideoElement, video: WorkbenchVideo) {
  const width = element.clientWidth;
  const height = element.clientHeight;
  const naturalWidth = element.videoWidth || video.width;
  const naturalHeight = element.videoHeight || video.height;
  const fit = getComputedStyle(element).objectFit;
  let drawnWidth = width;
  let drawnHeight = height;
  if (fit !== "fill") {
    const pick = fit === "cover" ? Math.max : Math.min;
    const scale = pick(width / naturalWidth, height / naturalHeight);
    drawnWidth = naturalWidth * scale;
    drawnHeight = naturalHeight * scale;
  }
  // Stored boxes are in the clip's own pixel space, whatever size was served.
  return {
    left: (width - drawnWidth) / 2,
    top: (height - drawnHeight) / 2,
    scaleX: drawnWidth / video.width,
    scaleY: drawnHeight / video.height
  };
}

/**
 * A source clip with the indexing run's tracks stroked over it as it plays.
 *
 * These are the same tracks, with the same ids, as the Tracking table and the
 * person index, drawn from the stored per-frame detections rather than from an
 * archived render of a different run. Person Search and Tracking both use it,
 * so the two views can never disagree about who was on screen.
 */
export function TrackPlayer({
  video,
  settings,
  live,
  focusTrack = null,
  focusPerson = null,
  seekTo,
  onFrame,
  className
}: {
  video: WorkbenchVideo;
  settings: Settings;
  /** Index people that survive the current thresholds; only they are drawn green. */
  live: ReadonlySet<string>;
  /** A track to emphasise; the others are dimmed while it is set. */
  focusTrack?: number | null;
  /**
   * A person to emphasise. The index merges fragmented tracks into one person,
   * so this can light several track ids at once.
   */
  focusPerson?: string | null;
  /** Replaced whenever the caller asks the player to jump to a frame. */
  seekTo: { frame: number; nonce: number } | null;
  onFrame?: (frame: number, boxes: number) => void;
  className?: string;
}) {
  const player = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const lastFrame = useRef(-1);

  const draw = useCallback(
    (mediaTime: number) => {
      const element = player.current;
      const surface = canvas.current;
      if (!element || !surface) return;

      const width = element.clientWidth;
      const height = element.clientHeight;
      const ratio = window.devicePixelRatio || 1;
      if (surface.width !== Math.round(width * ratio) || surface.height !== Math.round(height * ratio)) {
        surface.width = Math.round(width * ratio);
        surface.height = Math.round(height * ratio);
      }
      const ctx = surface.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // The frame on screen, not the continuous clock: a box belongs to a frame.
      const frame = Math.min(video.frame_count - 1, Math.max(0, Math.floor(mediaTime * video.fps + 1e-3)));
      const boxes: TrackBox[] = trackBoxesForFrame(video.id, frame, settings);
      const box = contentBox(element, video);
      const focusing = focusTrack !== null || focusPerson !== null;

      ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textBaseline = "top";
      for (const track of boxes) {
        const rgb = track.personId && live.has(track.personId) ? INDEXED : UNINDEXED;
        const chosen =
          focusTrack === track.trackId || (focusPerson !== null && track.personId === focusPerson);
        const focused = !focusing || chosen;
        const alpha = focused ? 0.95 : 0.3;
        const x1 = box.left + track.bbox[0] * box.scaleX;
        const y1 = box.top + track.bbox[1] * box.scaleY;
        const x2 = box.left + track.bbox[2] * box.scaleX;
        const y2 = box.top + track.bbox[3] * box.scaleY;
        ctx.lineWidth = chosen ? 3 : 1.5;
        ctx.strokeStyle = `rgba(${rgb}, ${alpha})`;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `t${track.trackId}`;
        const labelWidth = ctx.measureText(label).width + 6;
        const labelY = y1 >= 14 ? y1 - 14 : y1;
        ctx.fillStyle = `rgba(${rgb}, ${alpha})`;
        ctx.fillRect(x1, labelY, labelWidth, 14);
        ctx.fillStyle = `rgba(0, 0, 0, ${focused ? 0.9 : 0.5})`;
        ctx.fillText(label, x1 + 3, labelY + 2);
      }

      if (frame !== lastFrame.current) {
        lastFrame.current = frame;
        onFrame?.(frame, boxes.length);
      }
    },
    [video, settings, live, focusTrack, focusPerson, onFrame]
  );

  // Follow playback frame by frame where the browser can say which frame is on
  // screen, and fall back to animation frames where it cannot.
  useEffect(() => {
    const element = player.current;
    if (!element) return;
    let handle = 0;
    let stopped = false;
    // Not every browser that runs this page has it, whatever the DOM types say.
    const perFrame = typeof element.requestVideoFrameCallback === "function";
    if (perFrame) {
      const step: VideoFrameRequestCallback = (_now, meta) => {
        if (stopped) return;
        draw(meta.mediaTime);
        handle = element.requestVideoFrameCallback(step);
      };
      handle = element.requestVideoFrameCallback(step);
    } else {
      const step = () => {
        if (stopped) return;
        draw(element.currentTime);
        handle = requestAnimationFrame(step);
      };
      handle = requestAnimationFrame(step);
    }

    // Paused, seeked or resized: there is no new video frame to wait for.
    const redraw = () => draw(element.currentTime);
    const events = ["seeked", "loadeddata", "loadedmetadata", "pause"];
    for (const name of events) element.addEventListener(name, redraw);
    const observer = new ResizeObserver(redraw);
    observer.observe(element);
    redraw();

    return () => {
      stopped = true;
      if (perFrame) element.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
      for (const name of events) element.removeEventListener(name, redraw);
      observer.disconnect();
    };
  }, [draw]);

  useEffect(() => {
    const element = player.current;
    if (!element || !seekTo) return;
    element.pause();
    element.currentTime = frameTime(seekTo.frame, video.fps);
  }, [seekTo, video.fps]);

  return (
    <div className={className ? `player ${className}` : "player"}>
      <video
        aria-label={`${video.label} with tracked boxes`}
        controls
        loop
        muted
        playsInline
        poster={video.poster}
        preload="auto"
        ref={player}
        src={video.source}
      />
      <canvas aria-hidden="true" className="player__overlay" ref={canvas} />
    </div>
  );
}
