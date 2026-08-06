import { ChevronDown, Play, RotateCcw, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

/* ---------------------------------------------------------------- sliders */

export type SliderSpec = {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
};

export function SettingsSlider({
  spec,
  value,
  onChange
}: {
  spec: SliderSpec;
  value: number;
  onChange: (next: number) => void;
}) {
  const decimals = spec.step < 1 ? String(spec.step).split(".")[1]?.length ?? 2 : 0;
  return (
    <label className="wb-slider">
      <span className="wb-slider__head">
        <span>{spec.label}</span>
        <output>{value.toFixed(decimals)}</output>
      </span>
      <input
        max={spec.max}
        min={spec.min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={spec.step}
        type="range"
        value={value}
      />
      <span className="wb-slider__scale">
        <span>{spec.min}</span>
        <span>{spec.max}</span>
      </span>
    </label>
  );
}

export function useSliders(specs: SliderSpec[]) {
  const [values, setValues] = useState(() => specs.map((spec) => spec.value));
  const set = useCallback((index: number, next: number) => {
    setValues((current) => current.map((value, i) => (i === index ? next : value)));
  }, []);
  const reset = useCallback(() => setValues(specs.map((spec) => spec.value)), [specs]);
  return { values, set, reset };
}

export function SettingsAccordion({
  title,
  specs,
  values,
  onChange,
  defaultOpen = true
}: {
  title: string;
  specs: SliderSpec[];
  values: number[];
  onChange: (index: number, next: number) => void;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="wb-accordion" data-open={open}>
      <button aria-expanded={open} onClick={() => setOpen((value) => !value)} type="button">
        <ChevronDown size={15} />
        {title}
      </button>
      {open ? (
        <div className="wb-accordion__body">
          {specs.map((spec, index) => (
            <SettingsSlider key={spec.label} onChange={(next) => onChange(index, next)} spec={spec} value={values[index]} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

/* ----------------------------------------------------------------- fields */

export function ModelPicker({
  value,
  onChange,
  options
}: {
  value: string;
  onChange: (next: string) => void;
  options: { key: string; label: string; available: boolean }[];
}) {
  return (
    <label className="wb-field">
      <span>Model</span>
      <select onChange={(event) => onChange(event.target.value)} value={value}>
        {options.map((option) => (
          <option disabled={!option.available} key={option.key} value={option.key}>
            {option.label}
            {option.available ? "" : " (missing)"}
          </option>
        ))}
      </select>
    </label>
  );
}

/** A video input rendered as though the visitor had already dropped a file in. */
export function VideoSlot({
  label,
  fileName,
  src,
  poster,
  optional = false
}: {
  label: string;
  fileName: string;
  src: string;
  poster: string;
  optional?: boolean;
}) {
  return (
    <figure className="wb-slot">
      <figcaption>
        <span>
          {label}
          {optional ? " optional" : ""}
        </span>
        <small title={fileName}>
          <Upload size={12} /> {fileName}
        </small>
      </figcaption>
      <video controls loop muted playsInline poster={poster} preload="none" src={src} />
    </figure>
  );
}

/* ------------------------------------------------------------ run button */

export type RunState = "idle" | "running" | "done";

type RunStep = { at: number; message: string };

export function useSimulatedRun(steps: RunStep[], totalMs = 2200) {
  const [state, setState] = useState<RunState>("idle");
  const [progress, setProgress] = useState(0);
  // setInterval rather than requestAnimationFrame: rAF is suspended in
  // background tabs, which would leave a started run stuck at 0% forever.
  const timer = useRef<number | null>(null);
  const started = useRef(0);

  const cancel = useCallback(() => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
  }, []);

  useEffect(() => cancel, [cancel]);

  const start = useCallback(() => {
    cancel();
    setState("running");
    setProgress(0);
    started.current = Date.now();
    timer.current = window.setInterval(() => {
      const ratio = Math.min(1, (Date.now() - started.current) / totalMs);
      setProgress(ratio);
      if (ratio >= 1) {
        cancel();
        setState("done");
      }
    }, 60);
  }, [cancel, totalMs]);

  const reset = useCallback(() => {
    cancel();
    setState("idle");
    setProgress(0);
  }, [cancel]);

  const message = steps.reduce(
    (current, step) => (progress >= step.at ? step.message : current),
    steps[0]?.message ?? "Working…"
  );

  return { state, progress, message, start, reset };
}

export function RunButton({
  label,
  runningLabel,
  run
}: {
  label: string;
  runningLabel: string;
  run: ReturnType<typeof useSimulatedRun>;
}) {
  return (
    <div className="wb-run">
      <button
        className="wb-run__button"
        disabled={run.state === "running"}
        onClick={run.state === "done" ? run.reset : run.start}
        type="button"
      >
        {run.state === "done" ? <RotateCcw size={16} /> : <Play size={16} />}
        {run.state === "running" ? runningLabel : run.state === "done" ? "Reset" : label}
      </button>
      {run.state !== "idle" ? (
        <div className="wb-run__progress" role="status">
          <div className="wb-run__bar">
            <span style={{ width: `${Math.round(run.progress * 100)}%` }} />
          </div>
          <small>
            {run.state === "done" ? "Complete" : run.message} · {Math.round(run.progress * 100)}%
          </small>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------- read-outs */

export function OutputBox({
  label,
  children,
  lines
}: {
  label: string;
  children: React.ReactNode;
  lines?: number;
}) {
  return (
    <section className="wb-output">
      <header>{label}</header>
      <div className="wb-output__body" style={lines ? { minHeight: `${lines * 1.5}em` } : undefined}>
        {children}
      </div>
    </section>
  );
}

export function Placeholder({ text }: { text: string }) {
  return <p className="wb-placeholder">{text}</p>;
}
