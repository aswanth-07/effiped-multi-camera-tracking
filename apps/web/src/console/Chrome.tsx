import { Activity, ChevronDown, Cpu, ExternalLink, SlidersHorizontal } from "lucide-react";

import type { RunResult } from "../engine/pipeline";
import { repository } from "../lib/site";

export type ViewId = "sources" | "detection" | "tracking" | "search" | "cross" | "model";

export const VIEWS: { id: ViewId; label: string; key: string }[] = [
  { id: "search", label: "Person Search", key: "1" },
  { id: "detection", label: "Detection", key: "2" },
  { id: "tracking", label: "Tracking", key: "3" },
  { id: "cross", label: "Cross Camera", key: "4" },
  { id: "sources", label: "Sources", key: "5" },
  { id: "model", label: "Model", key: "6" }
];

function Mark() {
  return (
    <svg aria-hidden="true" className="mark" viewBox="0 0 48 48">
      <rect height="47" width="47" x="0.5" y="0.5" />
      <path d="M12 33 L20.5 15 L27.5 33" />
      <path d="M15.6 27.4 H24.9" />
      <path d="M31 33 V15 H36 a5 5 0 0 1 0 10 H31" />
      <path d="M34.4 25 L38.5 33" />
    </svg>
  );
}

export function TopBar({
  view,
  onView,
  running,
  controlsOpen,
  onToggleControls,
  dirty
}: {
  view: ViewId;
  onView: (next: ViewId) => void;
  running: boolean;
  controlsOpen: boolean;
  onToggleControls: () => void;
  dirty: boolean;
}) {
  return (
    <header className="topbar">
      <div className="topbar__identity">
        <Mark />
        <div className="topbar__name">
          <strong>EffiPed</strong>
          <span>Identity Review Console</span>
        </div>
        <span className={running ? "chip chip--live" : "chip"}>
          <span className="chip__dot" aria-hidden="true" />
          {running ? "Running" : "Replay"}
        </span>
      </div>

      <nav aria-label="Workspace" className="topbar__nav" role="tablist">
        {VIEWS.map((item) => (
          <button
            aria-controls={`view-${item.id}`}
            aria-selected={view === item.id}
            className={view === item.id ? "tab is-on" : "tab"}
            id={`tab-${item.id}`}
            key={item.id}
            onClick={() => onView(item.id)}
            role="tab"
            type="button"
          >
            <span className="tab__label">{item.label}</span>
            <kbd className="tab__key">{item.key}</kbd>
          </button>
        ))}
      </nav>

      <div className="topbar__tools">
        <button
          aria-expanded={controlsOpen}
          className={controlsOpen ? "tool-button is-on" : "tool-button"}
          onClick={onToggleControls}
          type="button"
        >
          <SlidersHorizontal aria-hidden="true" size={13} />
          <span>Thresholds</span>
          {dirty ? <span className="tool-button__dot" aria-hidden="true" /> : null}
          <ChevronDown aria-hidden="true" className="tool-button__caret" size={12} />
        </button>
        <a className="topbar__link" href={repository} rel="noreferrer" target="_blank">
          <span>Source</span>
          <ExternalLink aria-hidden="true" size={11} />
        </a>
      </div>
    </header>
  );
}

function Cell({ label, value, previous, format }: {
  label: string;
  value: number;
  previous: number | null;
  format?: (n: number) => string;
}) {
  const shown = format ? format(value) : value.toLocaleString();
  const delta = previous === null ? 0 : value - previous;
  const moved = Math.abs(delta) > (format ? 0.0005 : 0);
  return (
    <span className="cell">
      <span className="cell__label">{label}</span>
      <b className="cell__value">{shown}</b>
      {moved ? (
        <span className={delta > 0 ? "cell__delta is-up" : "cell__delta is-down"}>
          {delta > 0 ? "+" : "-"}
          {format ? format(Math.abs(delta)) : Math.abs(delta).toLocaleString()}
        </span>
      ) : null}
    </span>
  );
}

export function StatusBar({
  result,
  previous,
  stage
}: {
  result: RunResult | null;
  previous: RunResult | null;
  stage: string;
}) {
  const s = result?.stats;
  const p = previous?.stats ?? null;
  const three = (n: number) => n.toFixed(3);

  return (
    <footer className="statusbar">
      <span className="statusbar__stage">
        <Activity aria-hidden="true" size={11} />
        {stage}
      </span>

      <span className="statusbar__cells">
        <Cell label="clips" previous={p?.clips ?? null} value={s?.clips ?? 0} />
        <Cell label="frames" previous={p?.framesProcessed ?? null} value={s?.framesProcessed ?? 0} />
        <Cell label="detections" previous={p?.detectionsKept ?? null} value={s?.detectionsKept ?? 0} />
        <Cell label="people" previous={p?.peopleIndexed ?? null} value={s?.peopleIndexed ?? 0} />
        <Cell label="links" previous={p?.crossLinks ?? null} value={s?.crossLinks ?? 0} />
        <Cell
          format={three}
          label="mean conf"
          previous={p?.meanConfidence ?? null}
          value={s?.meanConfidence ?? 0}
        />
      </span>

      <span className="statusbar__mode" title="No model runs in this browser">
        <Cpu aria-hidden="true" size={11} />
        no inference in browser
      </span>
    </footer>
  );
}

export function Skeleton({ kind }: { kind: ViewId }) {
  const rows = kind === "search" ? 12 : 6;
  return (
    <div aria-busy="true" aria-live="polite" className={`skeleton skeleton--${kind}`}>
      <span className="sr-only">Applying thresholds</span>
      {Array.from({ length: rows }, (_, index) => (
        <span className="skeleton__block" key={index} style={{ animationDelay: `${index * 40}ms` }} />
      ))}
    </div>
  );
}
