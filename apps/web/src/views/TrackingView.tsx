import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { frameTime } from "../components/frameGrabber";
import {
  trackBoxesForFrame,
  tracksFor,
  type RunResult,
  type Settings,
  type TrackBox
} from "../engine/pipeline";
import type { WorkbenchVideo } from "../types";

const INDEXED = "134, 255, 107";
const UNINDEXED = "210, 153, 34";

/**
 * The source clip with the indexing run's tracks stroked over it as it plays.
 *
 * These are the same tracks, with the same ids, as the table beside it and the
 * person index, drawn from the stored per-frame detections rather than from an
 * archived render of a different run.
 */
function TrackPlayer({
  video,
  settings,
  live,
  focus,
  seekTo,
  onFrame
}: {
  video: WorkbenchVideo;
  settings: Settings;
  /** Index people that survive the current thresholds; only they are drawn green. */
  live: ReadonlySet<string>;
  /** A track to emphasise; the others are dimmed while it is set. */
  focus: number | null;
  /** Replaced whenever the table asks the player to jump to a frame. */
  seekTo: { frame: number; nonce: number } | null;
  onFrame: (frame: number, boxes: number) => void;
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
      const scale = width / (element.videoWidth || video.width);

      ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textBaseline = "top";
      for (const box of boxes) {
        const rgb = box.personId && live.has(box.personId) ? INDEXED : UNINDEXED;
        const focused = focus === null || focus === box.trackId;
        const alpha = focused ? 0.95 : 0.3;
        const [x1, y1, x2, y2] = box.bbox.map((value) => value * scale);
        ctx.lineWidth = focus === box.trackId ? 3 : 1.5;
        ctx.strokeStyle = `rgba(${rgb}, ${alpha})`;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `t${box.trackId}`;
        const labelWidth = ctx.measureText(label).width + 6;
        const labelY = y1 >= 14 ? y1 - 14 : y1;
        ctx.fillStyle = `rgba(${rgb}, ${alpha})`;
        ctx.fillRect(x1, labelY, labelWidth, 14);
        ctx.fillStyle = `rgba(0, 0, 0, ${focused ? 0.9 : 0.5})`;
        ctx.fillText(label, x1 + 3, labelY + 2);
      }

      if (frame !== lastFrame.current) {
        lastFrame.current = frame;
        onFrame(frame, boxes.length);
      }
    },
    [video, settings, live, focus, onFrame]
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
    const events = ["seeked", "loadeddata", "pause"];
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
    <div className="player">
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

export function TrackingView({
  result,
  settings,
  onInspect
}: {
  result: RunResult | null;
  settings: Settings;
  onInspect: (personId: string) => void;
}) {
  const [clip, setClip] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [seekTo, setSeekTo] = useState<{ frame: number; nonce: number } | null>(null);
  const [onScreen, setOnScreen] = useState({ frame: 0, boxes: 0 });
  const sources = result?.sources ?? [];
  const activeClip = clip !== null && sources.some((s) => s.id === clip) ? clip : sources[0]?.id ?? null;
  const video = sources.find((s) => s.id === activeClip);

  const onFrame = useCallback((frame: number, boxes: number) => setOnScreen({ frame, boxes }), []);
  const live = useMemo(() => new Set(Object.keys(result?.byId ?? {})), [result]);

  if (!result || !video) {
    return <div className="empty"><p>Run the pipeline to see tracks.</p></div>;
  }

  const rows = tracksFor(video.id, settings, result);
  const indexed = new Set(rows.flatMap((t) => (t.person ? [t.person.person.id] : []))).size;
  const focus = hovered ?? (rows.some((row) => row.trackId === selected) ? selected : null);

  const choose = (trackId: number, firstFrame: number) => {
    if (selected === trackId) {
      setSelected(null);
      return;
    }
    setSelected(trackId);
    setSeekTo({ frame: firstFrame, nonce: Date.now() });
  };

  return (
    <div className="view view--tracking">
      <div className="view__head">
        <h2>Camera local tracking</h2>
        <div className="seg">
          {sources.map((source) => (
            <button
              aria-pressed={source.id === activeClip}
              className={source.id === activeClip ? "seg__item is-on" : "seg__item"}
              key={source.id}
              onClick={() => {
                setClip(source.id);
                setSelected(null);
                setSeekTo(null);
              }}
              type="button"
            >
              {source.label}
            </button>
          ))}
        </div>
      </div>

      <div className="split">
        <figure className="stage stage--flush">
          <TrackPlayer
            focus={focus}
            live={live}
            key={video.id}
            onFrame={onFrame}
            seekTo={seekTo}
            settings={settings}
            video={video}
          />
          <figcaption className="stage__hud stage__hud--static">
            <span>{video.label}</span>
            <span>frame {onScreen.frame}</span>
            <span className="is-accent">{onScreen.boxes} on screen</span>
            <span>{rows.length} tracks</span>
            <span>{indexed} searchable people</span>
          </figcaption>
        </figure>

        <div className="table-pane">
          <table className="grid-table">
            <thead>
              <tr>
                <th scope="col">Track</th>
                <th scope="col">Frames</th>
                <th className="is-num" scope="col">Detections</th>
                <th className="is-num" scope="col">Best score</th>
                <th scope="col">Person Search</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const person = row.person;
                const classes = [person ? "" : "is-muted", selected === row.trackId ? "is-on" : ""]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <tr
                    aria-selected={selected === row.trackId}
                    className={classes || undefined}
                    key={row.trackId}
                    onBlur={() => setHovered(null)}
                    onClick={() => choose(row.trackId, row.firstFrame)}
                    onFocus={() => setHovered(row.trackId)}
                    onKeyDown={(event) => {
                      if (event.target !== event.currentTarget) return;
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        choose(row.trackId, row.firstFrame);
                      }
                    }}
                    onMouseEnter={() => setHovered(row.trackId)}
                    onMouseLeave={() => setHovered(null)}
                    tabIndex={0}
                  >
                    <th scope="row">
                      <code>t{row.trackId}</code>
                    </th>
                    <td>
                      {row.firstFrame} to {row.lastFrame}
                    </td>
                    <td className="is-num">{row.detections}</td>
                    <td className="is-num">{row.peakScore.toFixed(3)}</td>
                    <td>
                      {person ? (
                        <button
                          className="link-button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onInspect(person.person.id);
                          }}
                          type="button"
                        >
                          Open,{" "}
                          {person.matches.length === 0
                            ? "no matches"
                            : `${person.matches.length} ${person.matches.length === 1 ? "match" : "matches"}`}
                        </button>
                      ) : row.personId ? (
                        "below threshold"
                      ) : (
                        "not searchable"
                      )}
                    </td>
                  </tr>
                );
              })}
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={5}>No track has a detection above the current thresholds.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
