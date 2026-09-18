import { useState } from "react";

import type { RunResult } from "../engine/pipeline";

export function TrackingView({
  result,
  onInspect
}: {
  result: RunResult | null;
  onInspect: (personId: string) => void;
}) {
  const [clip, setClip] = useState<number | null>(null);
  const sources = result?.sources ?? [];
  const activeClip = clip !== null && sources.some((s) => s.id === clip) ? clip : sources[0]?.id ?? null;
  const video = sources.find((s) => s.id === activeClip);

  if (!result || !video) {
    return <div className="empty"><p>Run the pipeline to build tracks.</p></div>;
  }

  const rows = result.people
    .filter((row) => row.person.video_index === video.id)
    .sort((a, b) => a.person.track_id - b.person.track_id);

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
              onClick={() => setClip(source.id)}
              type="button"
            >
              {source.label}
            </button>
          ))}
        </div>
      </div>

      <div className="split">
        <figure className="stage stage--flush">
          <video
            aria-label={`${video.label} tracked output`}
            controls
            loop
            muted
            playsInline
            poster={video.poster}
            preload="metadata"
            src={video.tracked}
          />
          <figcaption className="stage__hud stage__hud--static">
            <span>{video.label}</span>
            <span>BoT-SORT</span>
            <span className="is-accent">{rows.length} tracks</span>
          </figcaption>
        </figure>

        <div className="table-pane">
          <table className="grid-table">
            <thead>
              <tr>
                <th scope="col">Track</th>
                <th scope="col">Frames</th>
                <th className="is-num" scope="col">Views</th>
                <th className="is-num" scope="col">Peak</th>
                <th className="is-num" scope="col">Links</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.person.id} onClick={() => onInspect(row.person.id)} tabIndex={0}
                    onKeyDown={(e) => { if (e.key === "Enter") onInspect(row.person.id); }}>
                  <th scope="row">
                    <code>t{row.person.track_id}</code>
                  </th>
                  <td>
                    {row.person.first_frame} to {row.person.last_frame}
                  </td>
                  <td className="is-num">{row.detections.length}</td>
                  <td className="is-num">{row.peakScore.toFixed(3)}</td>
                  <td className="is-num">{row.matches.length}</td>
                </tr>
              ))}
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={5}>No track clears the current activation threshold.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>

      <p className="view__foot">
        The tracked render is the original pipeline's own output for this clip. The table is
        recomputed live: a track appears once its best surviving detection clears the activation
        threshold.
      </p>
    </div>
  );
}
