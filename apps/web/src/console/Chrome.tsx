import { Activity, CircleDot, Cpu, ExternalLink } from "lucide-react";

import type { RunResult } from "../engine/pipeline";
import { repository } from "../lib/site";

export type ViewId = "sources" | "detection" | "tracking" | "search" | "cross" | "model";

export const VIEWS: { id: ViewId; label: string; key: string }[] = [
  { id: "sources", label: "Sources", key: "1" },
  { id: "detection", label: "Detection", key: "2" },
  { id: "tracking", label: "Tracking", key: "3" },
  { id: "search", label: "Person Search", key: "4" },
  { id: "cross", label: "Cross Camera", key: "5" },
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
  running
}: {
  view: ViewId;
  onView: (next: ViewId) => void;
  running: boolean;
}) {
  return (
    <header className="topbar">
      <div className="topbar__brand">
        <Mark />
        <div>
          <strong>EffiPed</strong>
          <span>Identity Review Console</span>
        </div>
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
            {item.label}
          </button>
        ))}
      </nav>

      <div className="topbar__right">
        <span className={running ? "runpill is-live" : "runpill"}>
          <CircleDot aria-hidden="true" size={11} />
          {running ? "Running" : "Replay"}
        </span>
        <a
          className="topbar__link"
          href={repository}
          rel="noreferrer"
          target="_blank"
          title="Source repository"
        >
          Source <ExternalLink aria-hidden="true" size={11} />
        </a>
      </div>
    </header>
  );
}

export function StatusBar({ result, stage }: { result: RunResult | null; stage: string }) {
  const stats = result?.stats;
  return (
    <footer className="statusbar">
      <span className="statusbar__stage">
        <Activity aria-hidden="true" size={11} />
        {stage}
      </span>
      <span className="statusbar__cells">
        <span>
          clips <b>{stats?.clips ?? 0}</b>
        </span>
        <span>
          frames <b>{stats?.framesProcessed ?? 0}</b>
        </span>
        <span>
          detections <b>{stats?.detectionsKept ?? 0}</b>
        </span>
        <span>
          people <b>{stats?.peopleIndexed ?? 0}</b>
        </span>
        <span>
          links <b>{stats?.crossLinks ?? 0}</b>
        </span>
        <span>
          mean conf <b>{stats ? stats.meanConfidence.toFixed(3) : "0.000"}</b>
        </span>
        <span>
          descriptor <b>{stats?.descriptorDim ?? 256}D</b>
        </span>
      </span>
      <span className="statusbar__mode" title="No model runs in this browser">
        <Cpu aria-hidden="true" size={11} />
        no inference in browser
      </span>
    </footer>
  );
}
