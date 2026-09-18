import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, Clock, Search, Video } from "lucide-react";

import { drawFrame } from "../components/frameGrabber";
import { clipDuration, provenanceFor, type PersonRow, type RunResult, type Settings } from "../engine/pipeline";

/**
 * Scrolls a container with an exponential ease-out.
 *
 * Neither `scrollTo({behavior:"smooth"})` nor `scroll-behavior: smooth` moves
 * this container in every environment the console runs in, and a jump that
 * sometimes animates and sometimes teleports is worse than either. Driving the
 * tween here makes the motion the same everywhere, and reduced motion still
 * gets the instant jump it asks for.
 */
function scrollTo(container: HTMLElement, top: number, reduced: boolean): () => void {
  const start = container.scrollTop;
  const distance = top - start;
  if (reduced || Math.abs(distance) < 2) {
    container.scrollTop = top;
    return () => {};
  }
  const duration = Math.min(420, 220 + Math.abs(distance) * 0.25);
  const began = performance.now();
  let frame = 0;
  const step = (now: number) => {
    const progress = Math.min(1, (now - began) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    container.scrollTop = start + distance * eased;
    if (progress < 1) frame = requestAnimationFrame(step);
  };
  frame = requestAnimationFrame(step);
  return () => cancelAnimationFrame(frame);
}

function tone(score: number): string {
  if (score >= 0.66) return "is-high";
  if (score >= 0.6) return "is-mid";
  return "is-low";
}

/**
 * The full frame a candidate's crop came from, with the camera and the moment
 * beside it. A ranked crop on its own is unreviewable: the reviewer's real
 * question is where and when this was, and whether the box is on the right
 * person. So the frame is redrawn from the clip at that timestamp with the
 * stored box stroked on it.
 */
function MatchFrame({ personId, similarity }: { personId: string; similarity: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const provenance = useMemo(() => provenanceFor(personId), [personId]);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !provenance) return;
    let cancelled = false;
    setState("loading");
    drawFrame(element, provenance.video.source, provenance.appearance.time_s, {
      box: provenance.appearance.bbox,
      boxLabel: `t${provenance.appearance.track_id}`,
      dim: true,
      width: 960
    })
      .then(() => !cancelled && setState("ready"))
      .catch(() => !cancelled && setState("error"));
    return () => {
      cancelled = true;
    };
  }, [provenance]);

  if (!provenance) {
    return <p className="pane__empty">No stored frame for this candidate.</p>;
  }

  const { video, appearance } = provenance;
  return (
    <div className="frame">
      <div className="frame__stage">
        <canvas
          aria-label={`Full frame from ${video.label} at ${appearance.time_s.toFixed(1)} seconds, with the matched person boxed`}
          ref={canvas}
        />
        {state === "loading" ? <span className="frame__status">decoding frame</span> : null}
        {state === "error" ? <span className="frame__status is-warn">frame unavailable</span> : null}
      </div>

      <dl className="frame__meta">
        <div>
          <dt>Source</dt>
          <dd>{video.label}</dd>
        </div>
        <div>
          <dt>Clip</dt>
          <dd><code>{video.file_name}</code></dd>
        </div>
        <div>
          <dt>Timestamp</dt>
          <dd>{appearance.time_s.toFixed(2)}s</dd>
        </div>
        <div>
          <dt>Frame</dt>
          <dd>{appearance.frame_index}</dd>
        </div>
        <div>
          <dt>Track</dt>
          <dd><code>t{appearance.track_id}</code></dd>
        </div>
        <div>
          <dt>Detection</dt>
          <dd className={tone(appearance.score)}>{appearance.score.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Similarity</dt>
          <dd className="is-accent">{similarity.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Box</dt>
          <dd>
            {Math.round(appearance.bbox[2] - appearance.bbox[0])} x{" "}
            {Math.round(appearance.bbox[3] - appearance.bbox[1])}
          </dd>
        </div>
      </dl>
    </div>
  );
}

export function SearchView({
  result,
  settings,
  selected,
  onSelect,
  onOpenSources
}: {
  result: RunResult | null;
  settings: Settings;
  selected: string | null;
  onSelect: (personId: string) => void;
  onOpenSources: () => void;
}) {
  const [sourceId, setSourceId] = useState<number | null>(null);
  const [span, setSpan] = useState<[number, number] | null>(null);
  const [activeMatch, setActiveMatch] = useState<string | null>(null);
  const matchesRef = useRef<HTMLElement>(null);

  const sources = result?.sources ?? [];
  const source = sources.find((video) => video.id === sourceId) ?? sources[0] ?? null;
  const duration = source ? clipDuration(source.id) : 15;
  const [from, to] = span ?? [0, duration];
  const currentSourceId = source?.id;

  // A new source resets the window rather than carrying a filter the reviewer
  // set for a different camera.
  useEffect(() => {
    setSpan(null);
  }, [currentSourceId]);

  const gallery = useMemo(() => {
    if (!result || !source) return [] as PersonRow[];
    return result.people.filter(
      (row) =>
        row.person.video_index === source.id &&
        row.person.last_time_s >= from &&
        row.person.first_time_s <= to
    );
  }, [result, source, from, to]);

  const row = selected && result ? result.byId[selected] ?? null : null;
  const matches = row?.matches ?? [];
  const rowId = row?.person.id;

  // Opening a person moves the reviewer to the candidates, which is the next
  // thing they need and sits below the fold on any realistic viewport.
  //
  // scrollIntoView alone is not reliable here: the crops above and inside the
  // candidate list load lazily, and a smooth scroll gets cancelled the moment
  // that shifts the layout under it. So the scroll is deferred past the commit
  // and driven on the scroll container itself, which survives the reflow.
  useEffect(() => {
    if (!rowId) return;
    setActiveMatch(null);
    let stop = () => {};
    const frame = requestAnimationFrame(() => {
      const target = matchesRef.current;
      if (!target) return;

      // Which element actually scrolls depends on the breakpoint: the viewport
      // pane owns it at desk width, and the document owns it on a phone where
      // the console stacks. Scrolling the wrong one is a silent no-op.
      const pane = target.closest(".viewport") as HTMLElement | null;
      const container =
        pane && pane.scrollHeight > pane.clientHeight + 1
          ? pane
          : (document.scrollingElement as HTMLElement | null);
      if (!container) return;

      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const base =
        container === document.scrollingElement ? 0 : container.getBoundingClientRect().top;
      const top = target.getBoundingClientRect().top - base + container.scrollTop;
      stop = scrollTo(container, Math.max(0, top - 8), reduced);
    });
    return () => {
      cancelAnimationFrame(frame);
      stop();
    };
  }, [rowId]);

  if (!result) {
    return (
      <div className="empty">
        <Search aria-hidden="true" size={22} />
        <p>Run the pipeline to build the person index.</p>
      </div>
    );
  }

  if (!source) {
    return (
      <div className="empty">
        <Video aria-hidden="true" size={22} />
        <p>No clip is attached to this job.</p>
        <button className="primary-button" onClick={onOpenSources} type="button">
          Select video sources
        </button>
      </div>
    );
  }

  const inSource = result.people.filter((r) => r.person.video_index === source.id).length;
  const active = activeMatch ? matches.find((m) => m.person.id === activeMatch) : undefined;

  return (
    <div className="search">
      <div className="search__top">
        <section aria-labelledby="src-heading" className="pane pane--source">
          <div className="pane__head">
            <h2 className="pane__title" id="src-heading">Source</h2>
            <span className="pane__count">{source.width}x{source.height} &middot; {source.fps}fps</span>
          </div>

          <video
            aria-label={`${source.label} footage`}
            className="pane__video"
            controls
            loop
            muted
            playsInline
            poster={source.poster}
            preload="metadata"
            src={source.source}
          />

          <div aria-label="Select source clip" className="picker" role="group">
            {sources.map((video) => (
              <button
                aria-pressed={video.id === source.id}
                className={video.id === source.id ? "picker__item is-on" : "picker__item"}
                key={video.id}
                onClick={() => setSourceId(video.id)}
                type="button"
              >
                <img alt="" loading="lazy" src={video.poster} />
                <span className="picker__name">{video.label}</span>
                <span className="picker__n">
                  {result.people.filter((r) => r.person.video_index === video.id).length}
                </span>
              </button>
            ))}
          </div>

          <button className="ghost-button ghost-button--wide" onClick={onOpenSources} type="button">
            <Video aria-hidden="true" size={12} />
            Change attached clips
          </button>
        </section>

        <section aria-labelledby="gal-heading" className="pane pane--gallery">
          <div className="pane__head">
            <h2 className="pane__title" id="gal-heading">People in {source.label}</h2>
            <span className="pane__count">
              {gallery.length} of {inSource}
            </span>
          </div>

          <div className="timefilter">
            <Clock aria-hidden="true" className="timefilter__icon" size={12} />
            <div className="timefilter__leg">
              <label htmlFor="time-from">From</label>
              <input
                id="time-from"
                max={duration}
                min={0}
                onChange={(event) => setSpan([Math.min(Number(event.target.value), to), to])}
                step={0.5}
                type="range"
                value={from}
              />
              <output htmlFor="time-from">{from.toFixed(1)}s</output>
            </div>
            <div className="timefilter__leg">
              <label htmlFor="time-to">To</label>
              <input
                id="time-to"
                max={duration}
                min={0}
                onChange={(event) => setSpan([from, Math.max(Number(event.target.value), from)])}
                step={0.5}
                type="range"
                value={to}
              />
              <output htmlFor="time-to">{to.toFixed(1)}s</output>
            </div>
            <button
              className="ghost-button ghost-button--mini"
              disabled={from === 0 && to === duration}
              onClick={() => setSpan(null)}
              type="button"
            >
              Whole clip
            </button>
          </div>

          {gallery.length === 0 ? (
            <p className="pane__empty">
              No track in {source.label} is present between {from.toFixed(1)}s and {to.toFixed(1)}s at
              the current thresholds. Widen the window, or lower the activation floor.
            </p>
          ) : (
            <div aria-label={`People detected in ${source.label}`} className="gallery" role="listbox">
              {gallery.map((entry) => {
                const isOn = entry.person.id === selected;
                return (
                  <button
                    aria-selected={isOn}
                    className={isOn ? "tile is-on" : "tile"}
                    key={entry.person.id}
                    onClick={() => onSelect(entry.person.id)}
                    role="option"
                    type="button"
                  >
                    {entry.person.crop ? (
                      <img alt="" className="tile__crop" loading="lazy" src={entry.person.crop} />
                    ) : (
                      <span className="tile__crop tile__crop--none" />
                    )}
                    <span className="tile__meta">
                      <code>t{entry.person.track_id}</code>
                      <span className={`tile__score ${tone(entry.peakScore)}`}>
                        {entry.peakScore.toFixed(2)}
                      </span>
                    </span>
                    <span className="tile__sub">
                      {entry.person.first_time_s.toFixed(1)}s to {entry.person.last_time_s.toFixed(1)}s
                    </span>
                    <span className="tile__sub tile__sub--links">
                      {entry.matches.length} candidate{entry.matches.length === 1 ? "" : "s"}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </section>
      </div>

      <section aria-labelledby="matches-heading" className="matches" ref={matchesRef}>
        <div className="pane__head">
          <h2 className="pane__title" id="matches-heading">
            {row ? (
              <>Candidates for <code>t{row.person.track_id}</code> in {source.label}</>
            ) : (
              "Candidates"
            )}
          </h2>
          {row ? (
            <span className="pane__count">
              {matches.length} above {settings.similarity.toFixed(2)} similarity
            </span>
          ) : null}
        </div>

        {!row ? (
          <p className="matches__prompt">
            <ArrowDown aria-hidden="true" size={13} />
            Select anyone in the gallery above to see who they may match in the other cameras.
          </p>
        ) : matches.length === 0 ? (
          <p className="pane__empty">
            Nothing clears a similarity of {settings.similarity.toFixed(2)} for this track. Lower the
            match threshold to widen the candidate set.
          </p>
        ) : (
          <div className="matches__body">
            <ol aria-label="Ranked candidates" className="ranked">
              {matches.map((match, rank) => {
                const isOn = match.person.id === activeMatch;
                return (
                  <li key={match.person.id}>
                    <button
                      aria-pressed={isOn}
                      className={isOn ? "ranked__item is-on" : "ranked__item"}
                      onClick={() => setActiveMatch(match.person.id)}
                      type="button"
                    >
                      <span className="ranked__rank">{rank + 1}</span>
                      {match.person.crop ? (
                        <img alt="" className="ranked__crop" loading="lazy" src={match.person.crop} />
                      ) : (
                        <span className="ranked__crop ranked__crop--none" />
                      )}
                      <span className="ranked__body">
                        <span className="ranked__where">
                          Camera {match.person.video_index + 1} &middot; <code>t{match.person.track_id}</code>
                        </span>
                        <span className="ranked__bar">
                          <span style={{ width: `${Math.max(2, match.similarity * 100)}%` }} />
                        </span>
                        <span className="ranked__time">
                          {match.person.first_time_s.toFixed(1)}s to {match.person.last_time_s.toFixed(1)}s
                        </span>
                      </span>
                      <span className="ranked__score">{match.similarity.toFixed(3)}</span>
                    </button>
                  </li>
                );
              })}
            </ol>

            <div className="matches__frame">
              {active ? (
                <MatchFrame personId={active.person.id} similarity={active.similarity} />
              ) : (
                <p className="matches__prompt">
                  <ArrowDown aria-hidden="true" size={13} />
                  Select a candidate to see the full frame it was cropped from.
                </p>
              )}
              <p className="matches__caution">
                Ranked candidates are reviewable appearance evidence. They are not an identification,
                and the score is not calibrated: in this session non-matches reach 0.998 against
                matches at 0.999.
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
