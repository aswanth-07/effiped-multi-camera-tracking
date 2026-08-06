import { useEffect, useRef, useState } from "react";

import { drawFrame } from "./frameGrabber";

type FrameViewProps = {
  src: string;
  timeS: number;
  box?: [number, number, number, number];
  boxLabel?: string;
  width?: number;
  dim?: boolean;
  alt: string;
  className?: string;
};

/** A still frame pulled from a demo clip, with the subject boxed. */
export function FrameView({
  src,
  timeS,
  box,
  boxLabel,
  width = 960,
  dim = false,
  alt,
  className
}: FrameViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    setState("loading");
    drawFrame(canvas, src, timeS, { box, boxLabel, width, dim })
      .then(() => {
        if (!cancelled) setState("ready");
      })
      .catch(() => {
        if (!cancelled) setState("error");
      });
    return () => {
      cancelled = true;
    };
    // box is a tuple; join it so we don't re-render on identical values.
  }, [src, timeS, box?.join(","), boxLabel, width, dim]);

  return (
    <div className={`frame-view ${className ?? ""}`.trim()} data-state={state}>
      <canvas aria-label={alt} ref={canvasRef} role="img" />
      {state === "loading" ? <span className="frame-view__note">Decoding frame…</span> : null}
      {state === "error" ? <span className="frame-view__note">Frame unavailable</span> : null}
    </div>
  );
}
