import { useState } from "react";

import { personSearch, sessionStats, videoFor } from "../data/personSearch";
import { results } from "../data/results";
import { PersonSearchTab } from "./PersonSearchTab";
import {
  ModelPicker,
  OutputBox,
  Placeholder,
  RunButton,
  SettingsAccordion,
  VideoSlot,
  useSimulatedRun,
  useSliders,
  type SliderSpec
} from "./workbenchControls";

const TABS = [
  "Single Camera",
  "Cross Camera",
  "Person Search",
  "Image Detection",
  "Model Status",
  "Research Context"
] as const;

type Tab = (typeof TABS)[number];

const MODELS = [
  { key: "effiped_tier1", label: "EffiPed Tier-1 fold-0", available: true },
  { key: "effiped_center", label: "EffiPed Center baseline fold-0", available: true },
  { key: "effiped_tier1_fold1", label: "EffiPed Tier-1 fold-1", available: false },
  { key: "effiped_part", label: "EffiPed Part-readout fold-0", available: false }
];

const singleSliders: SliderSpec[] = [
  { label: "Decode threshold", min: 0.01, max: 0.5, step: 0.01, value: 0.05 },
  { label: "Track activation threshold", min: 0.1, max: 0.8, step: 0.05, value: 0.3 },
  { label: "Top-K detections", min: 50, max: 600, step: 50, value: 300 },
  { label: "Minimum box height", min: 4, max: 40, step: 2, value: 10 },
  { label: "Frame limit", min: 0, max: 1200, step: 50, value: 150 }
];

const crossSliders: SliderSpec[] = [
  { label: "Decode threshold", min: 0.01, max: 0.5, step: 0.01, value: 0.05 },
  { label: "Track activation threshold", min: 0.1, max: 0.8, step: 0.05, value: 0.35 },
  { label: "Cross-camera merge threshold", min: 0.1, max: 0.9, step: 0.05, value: 0.3 },
  { label: "Top-K detections", min: 50, max: 600, step: 50, value: 300 },
  { label: "Frame limit", min: 0, max: 800, step: 50, value: 150 }
];

const imageSliders: SliderSpec[] = [
  { label: "Decode threshold", min: 0.01, max: 0.7, step: 0.01, value: 0.05 },
  { label: "Top-K detections", min: 50, max: 600, step: 50, value: 300 },
  { label: "Minimum box height", min: 4, max: 40, step: 2, value: 10 }
];

const trackingSteps = [
  { at: 0, message: "Decoding clip" },
  { at: 0.3, message: "Running detection" },
  { at: 0.62, message: "Associating tracks" },
  { at: 0.88, message: "Rendering overlays" }
];

/* -------------------------------------------------------- Single Camera */

function SingleCameraTab({ modelKey, onModelChange }: { modelKey: string; onModelChange: (k: string) => void }) {
  const sliders = useSliders(singleSliders);
  const run = useSimulatedRun(trackingSteps, 2000);
  const [cameraIndex, setCameraIndex] = useState(0);
  const video = videoFor(cameraIndex);
  const done = run.state === "done";

  return (
    <div className="wb-tab">
      <div className="wb-row wb-row--split">
        <div className="wb-col wb-col--controls">
          <label className="wb-field">
            <span>Video</span>
            <select
              onChange={(event) => {
                setCameraIndex(Number(event.target.value));
                run.reset();
              }}
              value={cameraIndex}
            >
              {personSearch.videos.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.file_name} — {item.label}
                </option>
              ))}
            </select>
          </label>

          <VideoSlot
            fileName={video.file_name}
            label="Uploaded clip"
            poster={video.poster}
            src={video.source}
          />

          <ModelPicker onChange={onModelChange} options={MODELS} value={modelKey} />
          <SettingsAccordion
            onChange={sliders.set}
            specs={singleSliders}
            title="Runtime settings"
            values={sliders.values}
          />
          <RunButton label="Run single-camera tracking" run={run} runningLabel="Tracking…" />
        </div>

        <div className="wb-col wb-col--wide">
          <OutputBox label="Tracked output">
            {done ? (
              <video controls key={video.tracked} loop muted playsInline poster={video.poster} src={video.tracked} />
            ) : (
              <Placeholder text="Run tracking to render detector and tracker overlays for this clip." />
            )}
          </OutputBox>
          <OutputBox label="Run summary" lines={8}>
            {done ? (
              <pre>
{`Source: ${video.file_name} (${video.width}x${video.height} @ ${video.fps} FPS)
Frames processed: ${sessionStats.maxFrames}
Tracks initialised: ${sessionStats.trackCounts[cameraIndex]}
Camera MOTA: ${sessionStats.perCameraMota[cameraIndex]}
Detection threshold: ${sessionStats.detThresh}
Session: P-DESTRE ${sessionStats.session} · Seq${cameraIndex + 1}
Note: replay of a precomputed run; sliders do not re-execute inference.`}
              </pre>
            ) : (
              <Placeholder text="No run yet." />
            )}
          </OutputBox>
        </div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------- Cross Camera */

function CrossCameraTab({ modelKey, onModelChange }: { modelKey: string; onModelChange: (k: string) => void }) {
  const sliders = useSliders(crossSliders);
  const run = useSimulatedRun(trackingSteps, 2400);
  const done = run.state === "done";

  return (
    <div className="wb-tab">
      <div className="wb-row wb-row--split">
        <div className="wb-col wb-col--controls">
          <div className="wb-slot-grid">
            {personSearch.videos.map((video, index) => (
              <VideoSlot
                fileName={video.file_name}
                key={video.id}
                label={`Camera ${index + 1}`}
                optional={index > 1}
                poster={video.poster}
                src={video.source}
              />
            ))}
          </div>
          <ModelPicker onChange={onModelChange} options={MODELS} value={modelKey} />
          <SettingsAccordion
            onChange={sliders.set}
            specs={crossSliders}
            title="Runtime settings"
            values={sliders.values}
          />
          <RunButton label="Run cross-camera association" run={run} runningLabel="Associating…" />
        </div>

        <div className="wb-col wb-col--wide">
          <OutputBox label="All tracks">
            {done ? (
              <video controls loop muted playsInline poster="/media/pdestre/four-camera-tracking.jpg" src="/media/pdestre/multi-camera-tracking.webm" />
            ) : (
              <Placeholder text="Run association to replay all camera-local tracks." />
            )}
          </OutputBox>
          <OutputBox label="Cross-camera matches">
            {done ? (
              <video controls loop muted playsInline poster="/media/pdestre/cross-camera-matches.jpg" src="/media/pdestre/cross-camera-matches.webm" />
            ) : (
              <Placeholder text="Run association to replay only the cross-camera matched tracks." />
            )}
          </OutputBox>
          <OutputBox label="Run summary" lines={9}>
            {done ? (
              <>
                <pre>
{`Cameras: 4 · frames per camera: ${sessionStats.maxFrames}
Overall MOTA: ${sessionStats.overallMota}
Cross-camera IDs: ${sessionStats.crossCameraIds}
Correct / wrong matches: ${sessionStats.correctMatches} / ${sessionStats.wrongMatches}
Pairwise association precision: ${(sessionStats.precision * 100).toFixed(0)}%
Merge threshold: ${sessionStats.mergeThresh}`}
                </pre>
                <table className="wb-table">
                  <thead>
                    <tr><th>Pair</th><th>Correct</th><th>Wrong</th><th>Shared</th><th>Precision</th></tr>
                  </thead>
                  <tbody>
                    {sessionStats.perPair.map((row) => (
                      <tr key={row.pair}>
                        <td>{row.pair}</td>
                        <td>{row.correct}</td>
                        <td>{row.wrong}</td>
                        <td>{row.shared}</td>
                        <td>{(row.precision * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : (
              <Placeholder text="No run yet." />
            )}
          </OutputBox>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------ Image Detection */

function ImageDetectionTab({ modelKey, onModelChange }: { modelKey: string; onModelChange: (k: string) => void }) {
  const sliders = useSliders(imageSliders);
  const run = useSimulatedRun([{ at: 0, message: "Running detection" }], 1200);
  const done = run.state === "done";

  return (
    <div className="wb-tab">
      <div className="wb-row wb-row--split">
        <div className="wb-col wb-col--controls">
          <figure className="wb-slot">
            <figcaption>
              <span>Image</span>
              <small>frame 60 · cam1.mp4</small>
            </figcaption>
            <img alt="Input frame for detection" src={personSearch.detection.input} />
          </figure>
          <ModelPicker onChange={onModelChange} options={MODELS} value={modelKey} />
          <SettingsAccordion
            onChange={sliders.set}
            specs={imageSliders}
            title="Runtime settings"
            values={sliders.values}
          />
          <RunButton label="Run detection" run={run} runningLabel="Detecting…" />
        </div>

        <div className="wb-col wb-col--wide">
          <OutputBox label="Detection output">
            {done ? (
              <img alt="Detection output with boxes" className="wb-image-out" src={personSearch.detection.output} />
            ) : (
              <Placeholder text="Run detection to overlay person boxes on this frame." />
            )}
          </OutputBox>
          <OutputBox label="Run summary" lines={6}>
            {done ? (
              <pre>
{`Input: 960x540 frame from cam1.mp4 at 6.0s
Decode threshold: ${sliders.values[0].toFixed(2)} (run used ${sessionStats.detThresh})
Minimum box height: ${sliders.values[2]} px
Session: P-DESTRE ${sessionStats.session} · Seq1
Note: replay of a precomputed detection pass.`}
              </pre>
            ) : (
              <Placeholder text="No run yet." />
            )}
          </OutputBox>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------- Model Status */

function ModelStatusTab() {
  const [stamp, setStamp] = useState(() => new Date().toISOString().replace("T", " ").slice(0, 19));
  return (
    <div className="wb-tab">
      <button className="wb-run__button" onClick={() => setStamp(new Date().toISOString().replace("T", " ").slice(0, 19))} type="button">
        Refresh model status
      </button>
      <OutputBox label="Runtime" lines={16}>
        <pre>
{`Runtime device: hosted static replay (no GPU attached)
Checked: ${stamp} UTC

${MODELS.map((model) => `${model.label}: ${model.available ? "ready" : "missing checkpoint"}`).join("\n")}

Published weights: ${results.system_benchmarks.model}
Weights status: withheld pending dataset rights review
Descriptor: ${results.system_benchmarks.footprint.descriptor_dim}-D part-aware embedding
Input resolution: ${results.system_benchmarks.footprint.input_resolution}
Parameters: ${results.system_benchmarks.footprint.parameters_m}M
Pipeline throughput: ~${results.system_benchmarks.footprint.pipeline_fps_approx} FPS

Person-search index in this build was computed with:
  ${personSearch.generated_with.checkpoint}
  frame stride ${personSearch.generated_with.settings.frame_stride}, ${personSearch.job.people_count} people indexed`}
        </pre>
      </OutputBox>
    </div>
  );
}

/* ------------------------------------------------------ Research Context */

function ResearchContextTab() {
  const benchmark = results.system_benchmarks;
  return (
    <div className="wb-tab wb-prose">
      <h3>What this application demonstrates</h3>
      <p>
        A single visual backbone drives detection, tracking, and identity retrieval. ConvNeXt V2 features
        feed CenterNet detection heads and a part-aware RoI descriptor; BoT-SORT maintains camera-local
        tracks, and the identity gallery ranks evidence across cameras.
      </p>
      <h3>Reported protocols</h3>
      <ul>
        <li>
          <strong>Detection</strong> — {benchmark.pdestre.validation.detection_map50}% mAP@0.5 on P-DESTRE
          validation ({benchmark.pdestre.validation.protocol}).
        </li>
        <li>
          <strong>Tracking</strong> — {benchmark.mot17.mota.toFixed(2)} MOTA, {benchmark.mot17.idf1} IDF1,
          {" "}{benchmark.mot17.hota} HOTA on MOT17 val-half ({benchmark.mot17.protocol}).
        </li>
        <li>
          <strong>Cross-camera retrieval</strong> — {benchmark.pdestre.validation.rank1_cross}% Rank-1 on
          validation, {benchmark.pdestre.test.rank1_cross}% on the held-out test split.
        </li>
      </ul>
      <h3>Scope of this replay</h3>
      <p>
        Every panel here replays a precomputed run over four P-DESTRE session {sessionStats.session}
        {" "}clips. The hosted build performs no inference: the controls are live, but changing them does
        not re-execute the model. Application metrics shown are demonstration diagnostics, not official
        evaluation numbers — those come from the reproducible evaluator in the research repository.
      </p>
      <p className="wb-provenance">{personSearch.generated_with.note}</p>
    </div>
  );
}

/* ---------------------------------------------------------------- shell */

export function Workbench() {
  const [tab, setTab] = useState<Tab>("Person Search");
  const [modelKey, setModelKey] = useState(MODELS[0].key);

  return (
    <section className="workbench" id="demo">
      <header className="wb-header">
        <div>
          <h2>EffiPed Pedestrian Tracker</h2>
          <p>
            Paper-aligned prototype for efficient JDE detection, tracking, and cross-camera identity
            association.
          </p>
        </div>
        <div className="wb-chips">
          <div><span>Main model</span><strong>EffiPed Tier-1</strong></div>
          <div><span>Baseline</span><strong>Center-sampled JDE</strong></div>
          <div><span>Primary protocol</span><strong>P-DESTRE per-date</strong></div>
          <div><span>Runtime source</span><strong>src/effiped</strong></div>
        </div>
      </header>

      <nav aria-label="Workbench tabs" className="wb-tabs" role="tablist">
        {TABS.map((name) => (
          <button
            aria-selected={tab === name}
            className={tab === name ? "active" : ""}
            key={name}
            onClick={() => setTab(name)}
            role="tab"
            type="button"
          >
            {name}
          </button>
        ))}
      </nav>

      <div className="wb-panel" role="tabpanel">
        {tab === "Single Camera" ? <SingleCameraTab modelKey={modelKey} onModelChange={setModelKey} /> : null}
        {tab === "Cross Camera" ? <CrossCameraTab modelKey={modelKey} onModelChange={setModelKey} /> : null}
        {tab === "Person Search" ? (
          <PersonSearchTab modelKey={modelKey} modelOptions={MODELS} onModelChange={setModelKey} />
        ) : null}
        {tab === "Image Detection" ? <ImageDetectionTab modelKey={modelKey} onModelChange={setModelKey} /> : null}
        {tab === "Model Status" ? <ModelStatusTab /> : null}
        {tab === "Research Context" ? <ResearchContextTab /> : null}
      </div>
    </section>
  );
}
