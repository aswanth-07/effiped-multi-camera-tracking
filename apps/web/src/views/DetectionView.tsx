import { useEffect, useMemo, useRef, useState } from "react";

import { drawFrame, frameTime, type Overlay } from "../components/frameGrabber";
import { detectionFrames, detectionsAt, type RunResult, type Settings } from "../engine/pipeline";

/**
 * Every box the detector returned is stored with its frame index, so a frame
 * can be reconstructed exactly: seek the clip to that timestamp and stroke every
 * detection that survives the current thresholds. Boxes whose track made it
 * into the person index are marked, so the gap between "detected" and
 * "indexed" is visible rather than hidden.
 */
export function DetectionView({ result, settings }: { result: RunResult | null; settings: Settings }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [clip, setClip] = useState<number | null>(null);
  const [frame, setFrame] = useState(0);
  const [status, setStatus] = useState("");

  const sources = result?.sources ?? [];
  const activeClip = clip !== null && sources.some((s) => s.id === clip) ? clip : sources[0]?.id ?? null;

  // Processed frames that still hold a detection for this clip.
  const frames = useMemo(
    () => (activeClip === null ? [] : detectionFrames(activeClip, settings)),
    [activeClip, settings]
  );

  const frameIndex = frames.includes(frame) ? frame : frames[0] ?? 0;

  const detections = useMemo(
    () => (activeClip === null ? [] : detectionsAt(activeClip, frameIndex, settings, result)),
    [activeClip, frameIndex, settings, result]
  );
  const indexed = detections.filter((detection) => detection.personId !== null).length;

  const overlays: Overlay[] = useMemo(
    () =>
      detections.map((detection) => ({
        box: detection.bbox,
        label: `${detection.trackId === null ? "" : `t${detection.trackId} `}${detection.score.toFixed(2)}`,
        color: detection.personId !== null ? "rgba(134, 255, 107, 0.95)" : "rgba(210, 153, 34, 0.95)"
      })),
    [detections]
  );

  const video = sources.find((s) => s.id === activeClip);
  const timeS = video ? frameIndex / video.fps : 0;

  useEffect(() => {
    const element = canvas.current;
    if (!element || !video) return;
    let cancelled = false;
    setStatus("decoding frame");
    drawFrame(element, video.source, frameTime(frameIndex, video.fps), { overlays, width: 960 })
      .then(() => !cancelled && setStatus(""))
      .catch(() => !cancelled && setStatus("frame unavailable"));
    return () => {
      cancelled = true;
    };
  }, [video, frameIndex, overlays]);

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
          <span className="is-accent">{detections.length} detections</span>
          <span>{indexed} indexed</span>
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
        Every box the detector returned on this frame, redrawn at a confidence floor of{" "}
        <b>{settings.detConf.toFixed(2)}</b> and a minimum height of <b>{settings.minBoxHeight}px</b>.
        Green boxes belong to a person in the search index at the current thresholds; amber ones were
        detected but are not part of it. Frames are sampled every second frame, as in the indexing run.
      </p>
    </div>
  );
}
