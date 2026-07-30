import {
  ArrowRight,
  Boxes,
  Camera,
  CheckCircle2,
  CodeXml,
  ExternalLink,
  Gauge,
  Network,
  ScanSearch,
  ShieldCheck
} from "lucide-react";

import { Architecture } from "./components/Architecture";
import { DemoConsole } from "./components/DemoConsole";
import { LiveConsole } from "./components/LiveConsole";
import { results } from "./data/results";

const repository = "https://github.com/aswanth-07/effiped-multi-camera-tracking";
const boxjdeRepository = "https://github.com/aswanth-07/boxjde-person-search";

function Benchmark({
  value,
  label,
  note
}: {
  value: string;
  label: string;
  note: string;
}) {
  return (
    <article className="benchmark-card">
      <strong>{value}</strong>
      <span>{label}</span>
      <small>{note}</small>
    </article>
  );
}

function App() {
  const benchmark = results.system_benchmarks;
  const mode = import.meta.env.VITE_APP_MODE ?? "demo";

  return (
    <div className="site-shell">
      <nav className="site-nav" aria-label="Primary navigation">
        <a className="site-brand" href="#top" aria-label="EffiPed home">
          <span>E</span>
          <strong>EffiPed</strong>
          <small>Identity Review</small>
        </a>
        <div>
          <a href="#demo">Demo app</a>
          <a href="#architecture">Architecture</a>
          <a href="#benchmarks">Benchmarks</a>
          <a href={repository} target="_blank" rel="noreferrer">
            <CodeXml size={15} /> GitHub
          </a>
        </div>
      </nav>

      <main id="top">
        <header className="project-hero">
          <div>
            <p className="hero-kicker">Multi-camera pedestrian intelligence</p>
            <h1>{results.project.title}</h1>
            <p className="hero-summary">{results.project.summary}</p>
            <div className="hero-actions">
              <a className="primary-link" href="#demo">
                Open the demo app <ArrowRight size={17} />
              </a>
              <a className="secondary-link" href={repository} target="_blank" rel="noreferrer">
                Browse the source <ExternalLink size={16} />
              </a>
            </div>
            <div className="project-byline">
              <span>Built by <strong>{results.project.author}</strong></span>
              <span>{mode === "demo" ? "Precomputed browser replay" : "Local API mode"}</span>
            </div>
          </div>
          <div className="hero-frame">
            <video
              aria-label="EffiPed four-camera tracking preview"
              autoPlay
              loop
              muted
              playsInline
              poster="/media/pdestre/four-camera-tracking.jpg"
              preload="metadata"
              src="/media/pdestre/multi-camera-tracking.webm"
            />
            <div className="hero-frame__hud">
              <span><span className="status-dot" /> Four synchronized views</span>
              <strong>Detector + tracker overlays from the original pipeline</strong>
            </div>
          </div>
        </header>

        {mode === "live" ? <LiveConsole /> : <DemoConsole />}

        <section className="content-section" id="architecture">
          <div className="section-heading">
            <span>01 / System design</span>
            <h2>One visual backbone supports detection, tracking, and identity retrieval.</h2>
            <p>
              ConvNeXt V2 features feed CenterNet detection and a part-aware RoI descriptor.
              BoT-SORT maintains local tracks; the identity gallery ranks evidence across cameras.
            </p>
          </div>
          <Architecture />
          <figure className="architecture-source">
            <img
              alt="EffiPed architecture from camera input through ConvNeXt V2, CenterNet, part-aware identity descriptors, tracking, and cross-camera review"
              loading="lazy"
              src="/architecture/effiped-architecture.svg"
            />
            <figcaption>End-to-end project architecture</figcaption>
          </figure>
        </section>

        <section className="content-section" id="benchmarks">
          <div className="section-heading">
            <span>02 / Measured system</span>
            <h2>Detection, tracking, retrieval, and efficiency are reported separately.</h2>
            <p>
              The cards below identify their evaluation protocol directly. The interactive replay
              is an application demonstration, not a benchmark run.
            </p>
          </div>
          <div className="benchmark-grid">
            <Benchmark
              value={`${benchmark.pdestre.validation.rank1_cross}%`}
              label="Cross-camera Rank-1"
              note="P-DESTRE validation · Protocol D"
            />
            <Benchmark
              value={`${benchmark.pdestre.test.rank1_cross}%`}
              label="Held-out Rank-1"
              note="P-DESTRE test · Protocol E"
            />
            <Benchmark
              value={benchmark.mot17.mota.toFixed(2)}
              label="MOT17 MOTA"
              note="Val-half · Protocol A"
            />
            <Benchmark
              value={`${benchmark.footprint.parameters_m}M`}
              label="Parameters"
              note={`≈${benchmark.footprint.pipeline_fps_approx} FPS · full pipeline`}
            />
          </div>
          <div className="protocol-grid">
            <article>
              <ScanSearch aria-hidden="true" />
              <span>Detection</span>
              <strong>{benchmark.pdestre.validation.detection_map50}% mAP@0.5</strong>
              <small>P-DESTRE validation</small>
            </article>
            <article>
              <Network aria-hidden="true" />
              <span>Tracking</span>
              <strong>{benchmark.mot17.idf1} IDF1 · {benchmark.mot17.hota} HOTA</strong>
              <small>MOT17 val-half</small>
            </article>
            <article>
              <Boxes aria-hidden="true" />
              <span>Identity descriptor</span>
              <strong>{benchmark.footprint.descriptor_dim}-D part-aware embedding</strong>
              <small>RoIAlign · four body strips · Coordinate Attention</small>
            </article>
            <article>
              <Gauge aria-hidden="true" />
              <span>Runtime</span>
              <strong>{benchmark.footprint.input_resolution} input</strong>
              <small>{benchmark.footprint.device}</small>
            </article>
          </div>
        </section>

        <section className="content-section">
          <div className="section-heading">
            <span>03 / Implementation</span>
            <h2>The hosted replay and local inference mode share the same review workflow.</h2>
          </div>
          <div className="engineering-grid">
            {[
              ["Perception", "ConvNeXt V2 · adaptive P2/P3 fusion · CenterNet heads", Camera],
              ["Identity", "RoIAlign · body strips · Coordinate Attention · 256-D descriptor", Boxes],
              ["Association", "BoT-SORT · Kalman motion · IoU and appearance matching", Network],
              ["Application", "React investigation console · FastAPI jobs · WebSocket progress", CodeXml],
              ["Review safety", "Candidate ranking with explicit human confirmation", ShieldCheck],
              ["Reproducibility", "Versioned configuration, evidence fixture, media hashes, CI", CheckCircle2]
            ].map(([title, copy, Icon]) => {
              const ItemIcon = Icon as typeof Camera;
              return (
                <article key={title as string}>
                  <ItemIcon aria-hidden="true" />
                  <strong>{title as string}</strong>
                  <p>{copy as string}</p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="content-section local-run">
          <div>
            <span>04 / Run locally</span>
            <h2>Use the hosted replay immediately, or connect the same UI to local GPU inference.</h2>
            <p>
              Model checkpoints are not bundled. Place an authorized checkpoint in
              <code> EFFIPED_WEIGHTS_DIR </code> before starting live mode.
            </p>
          </div>
          <pre><code>{`python -m venv .venv
pip install -e ".[runtime]"
effiped-app

# open http://127.0.0.1:8000`}</code></pre>
        </section>
      </main>

      <footer>
        <div>
          <strong>EffiPed Identity Review</strong>
          <span>Multi-camera pedestrian tracking and retrieval</span>
        </div>
        <p>
          Software © 2026 Aswanth Raj · Apache-2.0. P-DESTRE-derived media is
          separately licensed CC BY-NC-SA 4.0 for this non-commercial research demonstration.
        </p>
        <div>
          <a href={repository}>GitHub repository</a>
          <a href={boxjdeRepository}>Related BoxJDE research</a>
          <a href="/media/ASSET_MANIFEST.json">Media attribution</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
