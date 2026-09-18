import { AlertTriangle, Play, RotateCcw, Video } from "lucide-react";

import { belowIndexFloor, DEFAULT_SETTINGS, RANGES, type Settings } from "../engine/pipeline";

function Slider({
  range,
  value,
  onChange
}: {
  range: (typeof RANGES)[number];
  value: number;
  onChange: (next: number) => void;
}) {
  const id = `ctl-${range.key}`;
  const display =
    range.step < 1 ? value.toFixed(range.step < 0.01 ? 3 : 2) : `${value}${range.unit ?? ""}`;
  const pinned = range.indexedAt !== undefined && value < range.indexedAt;

  return (
    <div className="control">
      <div className="control__head">
        <label htmlFor={id}>{range.label}</label>
        <output className={pinned ? "control__value is-warn" : "control__value"} htmlFor={id}>
          {range.key === "frameLimit" && value === 0 ? "all" : display}
        </output>
      </div>
      <input
        aria-describedby={`${id}-hint`}
        className="control__range"
        id={id}
        max={range.max}
        min={range.min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={range.step}
        type="range"
        value={value}
      />
      <p className="control__hint" id={`${id}-hint`}>
        {range.hint}
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

  return (
    <aside aria-label="Pipeline controls" className="controls">
      <div className="controls__section">
        <p className="panel-label">Input</p>
        <button className="source-button" onClick={onOpenSources} type="button">
          <Video aria-hidden="true" size={14} />
          <span>Select video sources</span>
          <span className="source-button__count">{settings.sources.length}/4</span>
        </button>
      </div>

      <div className="controls__section controls__section--grow">
        <p className="panel-label">Thresholds</p>
        {RANGES.map((range) => (
          <Slider
            key={range.key}
            onChange={(next) => onChange({ ...settings, [range.key]: next })}
            range={range}
            value={settings[range.key] as number}
          />
        ))}
      </div>

      {floors.length > 0 ? (
        <p className="controls__warn" role="status">
          <AlertTriangle aria-hidden="true" size={13} />
          <span>
            {floors.map((range) => range.label).join(" and ")} sits below the floor this index was
            built at, so nothing new appears below it.
          </span>
        </p>
      ) : null}

      <div className="controls__actions">
        <button
          className="primary-button primary-button--run"
          disabled={running || noSources}
          onClick={onRun}
          type="button"
        >
          <Play aria-hidden="true" size={13} />
          {running ? "Running" : dirty ? "Run pipeline" : "Re-run"}
        </button>
        <button
          aria-label="Reset to defaults"
          className="icon-button"
          onClick={onReset}
          title="Reset to defaults"
          type="button"
        >
          <RotateCcw aria-hidden="true" size={14} />
        </button>
      </div>
      {noSources ? (
        <p className="controls__warn" role="status">
          <AlertTriangle aria-hidden="true" size={13} />
          <span>Attach at least one clip before running.</span>
        </p>
      ) : null}
      {dirty && !running && !noSources ? (
        <p className="controls__dirty" role="status">
          Settings changed since the last run.
        </p>
      ) : null}
    </aside>
  );
}

export { DEFAULT_SETTINGS };
