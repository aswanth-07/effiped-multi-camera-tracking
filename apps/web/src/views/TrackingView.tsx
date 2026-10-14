import { useCallback, useMemo, useState } from "react";

import { TrackPlayer } from "../components/TrackPlayer";
import { tracksFor, type RunResult, type Settings } from "../engine/pipeline";

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
            focusTrack={focus}
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
