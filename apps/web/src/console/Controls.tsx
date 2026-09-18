import { useMemo } from "react";
import { AlertTriangle, Play, RotateCcw, Video } from "lucide-react";

import {
  belowIndexFloor,
  DEFAULT_SETTINGS,
  DISTRIBUTIONS,
  RANGES,
  survivorsFor,
  type Range,
  type Settings
} from "../engine/pipeline";

/**
 * The histogram under each slider is the stored distribution the threshold cuts
 * through, drawn from the same records the run reads. Bars left of the cut are
 * discarded, bars right of it survive, so the cost of a move is visible before
 * it is made rather than only in the counters afterwards.
 */
function Histogram({ range, value }: { range: Range; value: number }) {
  const dist = DISTRIBUTIONS[range.key];
  if (!dist) return null;
  const span = range.max - range.min || 1;
  const cut = (value - range.min) / span;
  const keepsAbove = range.key !== "frameLimit";

  return (
    <div aria-hidden="true" className="histogram">
      {dist.bins.map((height, index) => {
        const at = index / (dist.bins.length - 1);
        const live = range.key === "frameLimit" && value === 0 ? true : keepsAbove ? at >= cut : at <= cut;
        return (
          <span
            className={live ? "histogram__bar is-live" : "histogram__bar"}
            key={index}
            style={{ height: `${Math.max(6, height * 100)}%` }}
          />
        );
      })}
    </div>
  );
}

function Control({
  range,
  value,
  onChange
}: {
  range: Range;
  value: number;
  onChange: (next: number) => void;
}) {
  const id = `ctl-${range.key}`;
  const survivors = useMemo(() => survivorsFor(range.key, value), [range.key, value]);
  const belowFloor = range.indexedAt !== undefined && value < range.indexedAt;

  const display =
    range.key === "frameLimit" && value === 0
      ? "all"
      : range.step < 1
        ? value.toFixed(range.step < 0.01 ? 3 : 2)
        : `${value}${range.unit ?? ""}`;

  return (
    <div className="control">
      <div className="control__head">
        <label htmlFor={id}>{range.label}</label>
        <output className={belowFloor ? "control__value is-warn" : "control__value"} htmlFor={id}>
          {display}
        </output>
      </div>

      <Histogram range={range} value={value} />

      <input
        aria-describedby={`${id}-note`}
        className="control__range"
        id={id}
        max={range.max}
        min={range.min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={range.step}
        type="range"
        value={value}
      />

      <p className="control__note" id={`${id}-note`}>
        {survivors ? (
          <span className="control__survivors">
            <b>{survivors.kept.toLocaleString()}</b> of {survivors.total.toLocaleString()} kept
          </span>
        ) : null}
        <span className="control__hint">{range.hint}</span>
      </p>
    </div>
  );
}

export function Controls({
  settings,
  dirty,
  running,
  onChange,
  onReset,
  onRun,
  onOpenSources
}: {
  settings: Settings;
  dirty: boolean;
  running: boolean;
  onChange: (next: Settings) => void;
  onReset: () => void;
  onRun: () => void;
  onOpenSources: () => void;
}) {
  const floors = belowIndexFloor(settings);
  const noSources = settings.sources.length === 0;
  const atDefaults = JSON.stringify(settings) === JSON.stringify(DEFAULT_SETTINGS);

  return (
    <aside aria-label="Pipeline controls" className="controls">
      <div className="controls__scroll">
        <section className="controls__section">
          <h2 className="panel-label">Input</h2>
          <button className="source-button" onClick={onOpenSources} type="button">
            <Video aria-hidden="true" size={14} />
            <span>Select video sources</span>
            <span className="source-button__count">{settings.sources.length}/4</span>
          </button>
        </section>

        <section className="controls__section">
          <h2 className="panel-label">Thresholds</h2>
          {RANGES.map((range) => (
            <Control
              key={range.key}
              onChange={(next) => onChange({ ...settings, [range.key]: next })}
              range={range}
              value={settings[range.key] as number}
            />
          ))}
        </section>
      </div>

      <div className="controls__dock">
        {floors.length > 0 ? (
          <p className="notice notice--warn" role="status">
            <AlertTriangle aria-hidden="true" size={13} />
            <span>
              {floors.map((range) => range.label).join(" and ")} sits below the floor this index was
              built at. Nothing new appears below it.
            </span>
          </p>
        ) : null}

        {noSources ? (
          <p className="notice notice--warn" role="status">
            <AlertTriangle aria-hidden="true" size={13} />
            <span>Attach at least one clip before running.</span>
          </p>
        ) : null}

        <div className="controls__actions">
          <button
            className="run-button"
            disabled={running || noSources}
            onClick={onRun}
            type="button"
          >
            <Play aria-hidden="true" size={13} />
            <span>{running ? "Running" : dirty ? "Run pipeline" : "Re-run"}</span>
            {dirty && !running && !noSources ? <span className="run-button__dot" aria-hidden="true" /> : null}
          </button>
          <button
            aria-label="Reset thresholds to defaults"
            className="icon-button"
            disabled={atDefaults}
            onClick={onReset}
            title="Reset thresholds to defaults"
            type="button"
          >
            <RotateCcw aria-hidden="true" size={14} />
          </button>
        </div>

        <p className="controls__state" role="status">
          {running
            ? "Applying thresholds"
            : dirty && !noSources
              ? "Settings changed since the last run"
              : "Output matches the current settings"}
        </p>
      </div>
    </aside>
  );
}

export { DEFAULT_SETTINGS };
