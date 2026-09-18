import { useEffect, useMemo, useRef, useState } from "react";

import { drawFrame, type Overlay } from "../components/frameGrabber";
import type { RunResult, Settings } from "../engine/pipeline";

/**
 * Detections are stored per appearance with a frame index and a box, so a frame
 * can be reconstructed exactly: seek the clip to that timestamp and stroke every
 * detection whose frame survived the current thresholds.
 */
export function DetectionView({ result, settings }: { result: RunResult | null; settings: Settings }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [clip, setClip] = useState<number | null>(null);
  const [frame, setFrame] = useState(0);
  const [status, setStatus] = useState("");

  const sources = result?.sources ?? [];
  const activeClip = clip !== null && sources.some((s) => s.id === clip) ? clip : sources[0]?.id ?? null;

  // Frames that actually hold a surviving detection for this clip.
  const frames = useMemo(() => {
    if (!result || activeClip === null) return [];
    const set = new Set<number>();
    for (const row of result.people) {
      if (row.person.video_index !== activeClip) continue;
      for (const detection of row.detections) set.add(detection.frame_index);
    }
    return [...set].sort((a, b) => a - b);
  }, [result, activeClip]);

  const frameIndex = frames.includes(frame) ? frame : frames[0] ?? 0;

  const overlays: Overlay[] = useMemo(() => {
    if (!result || activeClip === null) return [];
    const list: Overlay[] = [];
    for (const row of result.people) {
      if (row.person.video_index !== activeClip) continue;
      for (const detection of row.detections) {
        if (detection.frame_index !== frameIndex) continue;
        list.push({
          box: detection.bbox,
          label: `t${row.person.track_id} ${detection.score.toFixed(2)}`,
          color: detection.score >= 0.6 ? "rgba(134, 255, 107, 0.95)" : "rgba(210, 153, 34, 0.95)"
        });
      }
    }
    return list;
  }, [result, activeClip, frameIndex]);

  const video = sources.find((s) => s.id === activeClip);
  const timeS = video ? frameIndex / video.fps : 0;

  useEffect(() => {
    const element = canvas.current;
    if (!element || !video) return;
    let cancelled = false;
    setStatus("decoding frame");
    drawFrame(element, video.source, timeS, { overlays, width: 960 })
      .then(() => !cancelled && setStatus(""))
      .catch(() => !cancelled && setStatus("frame unavailable"));
    return () => {
      cancelled = true;
    };
  }, [video, timeS, overlays]);

  if (!result || sources.length === 0) {
    return <div className="empty"><p>Run the pipeline to decode detections.</p></div>;
  }

  return (
    <div className="view view--detection">
      <div className="view__head">
        <h2>Single frame detection</h2>
        <div className="seg">
          {sources.map((source) => (
            <button
              aria-pressed={source.id === activeClip}
              className={source.id === activeClip ? "seg__item is-on" : "seg__item"}
              key={source.id}
              onClick={() => {
                setClip(source.id);
                setFrame(0);
              }}
              type="button"
            >
              {source.label}
            </button>
          ))}
        </div>
      </div>

      <div className="stage">
        <canvas aria-label="Decoded frame with detection boxes" className="stage__canvas" ref={canvas} />
        {status ? <span className="stage__status">{status}</span> : null}
        <div className="stage__hud">
          <span>frame {frameIndex}</span>
          <span>t {timeS.toFixed(2)}s</span>
          <span className="is-accent">{overlays.length} detections</span>
        </div>
      </div>

      <div className="scrubber">
        <label htmlFor="frame-scrub">Frame with detections</label>
        <input
          disabled={frames.length === 0}
          id="frame-scrub"
          max={Math.max(0, frames.length - 1)}
          min={0}
          onChange={(event) => setFrame(frames[Number(event.target.value)] ?? 0)}
          step={1}
          type="range"
          value={Math.max(0, frames.indexOf(frameIndex))}
        />
        <span className="scrubber__count">
          {frames.length === 0 ? "none at these thresholds" : `${frames.indexOf(frameIndex) + 1} / ${frames.length}`}
        </span>
      </div>

      <p className="view__foot">
        Boxes are the stored detections for this frame, redrawn at a confidence floor of{" "}
        <b>{settings.detConf.toFixed(2)}</b> and a minimum height of <b>{settings.minBoxHeight}px</b>.
        Amber marks a detection under 0.60.
      </p>
    </div>
  );
}
