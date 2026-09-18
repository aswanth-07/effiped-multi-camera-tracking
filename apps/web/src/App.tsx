import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Controls } from "./console/Controls";
import { Skeleton, StatusBar, TopBar, VIEWS, type ViewId } from "./console/Chrome";
import { SourceModal } from "./console/SourceModal";
import { DEFAULT_SETTINGS, runQuery, type RunResult, type Settings } from "./engine/pipeline";
import { CrossCameraView } from "./views/CrossCameraView";
import { DetectionView } from "./views/DetectionView";
import { ModelView } from "./views/ModelView";
import { SearchView } from "./views/SearchView";
import { SourcesView } from "./views/SourcesView";
import { TrackingView } from "./views/TrackingView";

/**
 * The stages a run reports. They name what the replay is really doing, because
 * claiming to detect when the detections are already on disk would be a lie the
 * whole project is built to avoid.
 */
const STAGES = [
  "attaching clips",
  "reading stored detections",
  "applying confidence and size floors",
  "promoting tracks",
  "ranking cross-camera candidates",
  "ready"
];

export default function App() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [applied, setApplied] = useState<Settings | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [previous, setPrevious] = useState<RunResult | null>(null);
  const [view, setView] = useState<ViewId>("search");
  const [selected, setSelected] = useState<string | null>(null);
  const [modal, setModal] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [stageIndex, setStageIndex] = useState(-1);
  const timers = useRef<number[]>([]);
  /** The run the status bar compares against, kept out of render state. */
  const latest = useRef<RunResult | null>(null);

  const running = stageIndex >= 0 && stageIndex < STAGES.length - 1;
  const dirty = applied === null || JSON.stringify(applied) !== JSON.stringify(settings);

  const clearTimers = useCallback(() => {
    for (const id of timers.current) window.clearTimeout(id);
    timers.current = [];
  }, []);

  const run = useCallback(
    (next: Settings) => {
      clearTimers();
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const step = reduced ? 0 : 130;

      STAGES.forEach((_, index) => {
        timers.current.push(
          window.setTimeout(() => {
            setStageIndex(index);
            if (index === STAGES.length - 1) {
              const computed = runQuery(next);
              setPrevious(latest.current);
              latest.current = computed;
              setResult(computed);
              setApplied(next);
              // Keep the open selection when it survived the new thresholds.
              // A selection that no longer survives is dropped rather than
              // silently replaced with somebody else.
              setSelected((current) => (current && computed.byId[current] ? current : null));
            }
          }, index * step)
        );
      });
    },
    [clearTimers]
  );

  // First paint runs the default job so the console is never an empty shell.
  useEffect(() => {
    run(DEFAULT_SETTINGS);
    return clearTimers;
  }, [run, clearTimers]);

  // Number keys switch workspace, the way a tool does.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.key === "Escape") {
        setControlsOpen(false);
        return;
      }
      const match = VIEWS.find((item) => item.key === event.key);
      if (match) setView(match.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const inspect = useCallback((personId: string) => {
    setSelected(personId);
    setView("search");
  }, []);

  const stage = useMemo(() => {
    if (stageIndex < 0) return "idle";
    return STAGES[stageIndex];
  }, [stageIndex]);

  return (
    <div className="console">
      <TopBar
        controlsOpen={controlsOpen}
        dirty={dirty}
        onToggleControls={() => setControlsOpen((open) => !open)}
        onView={setView}
        running={running}
        view={view}
      />

      {controlsOpen ? (
        <>
          <Controls
            dirty={dirty}
            onChange={setSettings}
            onOpenSources={() => setModal(true)}
            onReset={() => setSettings(DEFAULT_SETTINGS)}
            onRun={() => run(settings)}
            running={running}
            settings={settings}
          />
          <button
            aria-label="Close thresholds"
            className="panel-scrim"
            onClick={() => setControlsOpen(false)}
            tabIndex={-1}
            type="button"
          />
        </>
      ) : null}

      <div className="workspace">
        <main aria-labelledby={`tab-${view}`} className="viewport" id={`view-${view}`} role="tabpanel">
          {running ? (
            <div className="runbar" role="status">
              <span className="runbar__spark" aria-hidden="true" />
              <span>{stage}</span>
            </div>
          ) : null}

          {result === null ? (
            <Skeleton kind={view} />
          ) : (
            <>
          {view === "sources" ? <SourcesView onOpenSources={() => setModal(true)} result={result} /> : null}
          {view === "detection" ? <DetectionView result={result} settings={applied ?? settings} /> : null}
          {view === "tracking" ? <TrackingView onInspect={inspect} result={result} /> : null}
          {view === "search" ? (
            <SearchView
              onOpenSources={() => setModal(true)}
              onSelect={setSelected}
              result={result}
              selected={selected}
              settings={applied ?? settings}
            />
          ) : null}
          {view === "cross" ? (
            <CrossCameraView onInspect={inspect} result={result} settings={applied ?? settings} />
          ) : null}
          {view === "model" ? <ModelView /> : null}
            </>
          )}
        </main>
      </div>

      <StatusBar previous={previous} result={result} stage={stage} />

      <SourceModal
        onCancel={() => setModal(false)}
        onConfirm={(ids) => {
          setModal(false);
          const next = { ...settings, sources: ids };
          setSettings(next);
          run(next);
        }}
        open={modal}
        selected={settings.sources}
      />
    </div>
  );
}
