import {
  ArrowDownRight,
  ArrowUpRight,
  Award,
  BookOpen,
  Box,
  Camera,
  CheckCircle2,
  CodeXml,
  ExternalLink,
  FileText,
  Layers3,
  ShieldAlert,
  Sparkles,
  Workflow
} from "lucide-react";
import { Architecture } from "./components/Architecture";
import { Investigation } from "./components/Investigation";
import { LiveConsole } from "./components/LiveConsole";
import { results } from "./data/results";

const repository = "https://github.com/aswanth-07/effiped-multi-camera-tracking";
const boxjdeRepository = "https://github.com/aswanth-07/boxjde-person-search";

function Metric({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <article className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
      <small>{note}</small>
    </article>
  );
}

function App() {
  const contest = results.verified_contest_system;
  const partjde = results.post_contest_evolution.partjde;
  const boxjde = results.post_contest_evolution.boxjde;
  const mode = import.meta.env.VITE_APP_MODE ?? "demo";

  return (
    <div className="site-shell">
      <nav className="nav">
        <a href="#top" className="brand" aria-label="EffiPed home">
          <span className="brand__mark">E</span>
          <span>EffiPed <em>/ SIPC 2026</em></span>
        </a>
        <div className="nav__links">
          <a href="#architecture">Architecture</a>
          <a href="#investigation">Investigation</a>
          <a href="#evidence">Evidence</a>
          <a href={repository} target="_blank" rel="noreferrer"><CodeXml size={16} /> Source</a>
        </div>
      </nav>

      <main>
        <section className="hero" id="top">
          <div className="hero__grid" aria-hidden="true" />
          <div className="hero__copy">
            <div className="award-line">
              <Award size={18} />
              <span>3rd Prize · Student Innovation Project Contest 2026</span>
            </div>
            <p className="eyebrow">Vertical 1 · AI & Intelligent Systems · VIT Vellore SCOPE</p>
            <h1>{results.project.title}</h1>
            <p className="hero__lede">
              One compact ConvNeXt V2 network detects pedestrians, maintains local tracks,
              and produces body-appearance descriptors for cross-camera identity review.
            </p>
            <div className="hero__actions">
              <a className="button button--primary" href="#investigation">
                Explore the investigation <ArrowDownRight size={18} />
              </a>
              <a className="button" href={repository} target="_blank" rel="noreferrer">
                View repository <ExternalLink size={17} />
              </a>
            </div>
            <div className="hero__credit">
              <span>Built by <strong>Aswanth Raj</strong></span>
              <span>Guide · <strong>Sri Preethaa KR</strong></span>
              <span className="mode-label">{mode === "demo" ? "Precomputed Vercel demo" : "Local live mode"}</span>
            </div>
          </div>

          <div className="hero__visual" aria-label="Four-camera system overview">
            {["C1", "C2", "C3", "C4"].map((camera, index) => (
              <div className={`hero-camera hero-camera--${index + 1}`} key={camera}>
                <span><Camera size={13} /> {camera}</span>
                <i className="person-box" />
              </div>
            ))}
            <div className="hero__core">
              <Sparkles size={21} />
              <strong>ONE NETWORK</strong>
              <small>detect · track · describe</small>
            </div>
            <div className="signal signal--a" />
            <div className="signal signal--b" />
          </div>
        </section>

        <section className="metric-strip" aria-label="Verified contest metrics">
          <Metric value={`${contest.pdestre.validation.rank1_cross}%`} label="Cross-camera Rank-1" note="P-DESTRE validation · Protocol D" />
          <Metric value={`${contest.pdestre.test.rank1_cross}%`} label="Held-out test Rank-1" note="P-DESTRE test · Protocol E" />
          <Metric value={contest.mot17.mota.toFixed(2)} label="MOT17 MOTA" note="Val-half · Protocol A" />
          <Metric value={`${contest.footprint.parameters_m}M`} label="Parameters" note={`≈${contest.footprint.pipeline_fps_approx} full-pipeline FPS`} />
        </section>

        <section className="section problem">
          <div className="section__intro">
            <span className="section-index">01</span>
            <div>
              <p className="eyebrow">The failure at the camera boundary</p>
              <h2>A person becomes a stranger every time the camera changes.</h2>
            </div>
          </div>
          <div className="problem__grid">
            <p className="display-copy">
              Traditional stacks run a detector, crop every person, then invoke a separate
              ReID model. EffiPed shares the expensive visual backbone and emits location
              and identity evidence in one pass.
            </p>
            <div className="comparison">
              <article>
                <span>Conventional stack</span>
                <strong>Detector → crop → ReID → tracker</strong>
                <small>Multiple networks and hand-off points</small>
              </article>
              <ArrowUpRight />
              <article className="comparison__active">
                <span>EffiPed</span>
                <strong>Frame → shared features → box + descriptor</strong>
                <small>One model, two coordinated tasks</small>
              </article>
            </div>
          </div>
        </section>

        <section className="section" id="architecture">
          <div className="section__intro">
            <span className="section-index">02</span>
            <div>
              <p className="eyebrow">End-to-end architecture</p>
              <h2>Detection finds the person. Part descriptors preserve what makes them distinct.</h2>
            </div>
          </div>
          <Architecture />
          <figure className="architecture-source">
            <img
              src="/architecture/effiped-architecture.svg"
              alt="Detailed EffiPed architecture from four camera inputs through ConvNeXt V2, P2/P3 fusion, CenterNet detection, four-strip identity descriptors, BoT-SORT, cross-camera candidate ranking, and analyst review."
            />
            <figcaption>Full contest pipeline · editable source retained in PowerPoint</figcaption>
          </figure>
          <div className="formula-row">
            <div><span>Part fusion</span><code>z = norm(Σ α<sub>k</sub> z<sub>k</sub>)</code></div>
            <div><span>Metric learning</span><code>L = log(1 + e<sup>d⁺ − d⁻</sup>)</code></div>
            <div><span>Angular classification</span><code>cos(θ<sub>y</sub> + m)</code></div>
          </div>
          <a className="text-link" href="/architecture/effiped-architecture.pptx">
            <FileText size={17} /> Download the editable PowerPoint architecture <ArrowUpRight size={16} />
          </a>
        </section>

        <section className="section investigation-section" id="investigation">
          <div className="section__intro">
            <span className="section-index">03</span>
            <div>
              <p className="eyebrow">Identity review, rebuilt for the web</p>
              <h2>Select a track, follow the handoff, and inspect the candidate evidence.</h2>
            </div>
          </div>
          {mode === "live" ? <LiveConsole /> : <Investigation />}
          <p className="license-note">
            Adapted P-DESTRE research media · CC BY-NC-SA 4.0 · precomputed demonstration · non-commercial use.
          </p>
        </section>

        <section className="section evidence" id="evidence">
          <div className="section__intro">
            <span className="section-index">04</span>
            <div>
              <p className="eyebrow">Evidence ledger</p>
              <h2>The contest result, the submitted poster, and the later research are kept separate.</h2>
            </div>
          </div>
          <div className="evidence__grid">
            <article className="evidence-card evidence-card--primary">
              <span className="tag">Verified contest system</span>
              <h3>EffiPed Tier-1</h3>
              <div className="evidence-number">{contest.pdestre.validation.rank1_cross}%</div>
              <p>Cross-camera Rank-1 on P-DESTRE fold-0 validation.</p>
              <dl>
                <div><dt>Held-out test</dt><dd>{contest.pdestre.test.rank1_cross}% R1</dd></div>
                <div><dt>Detection</dt><dd>{contest.pdestre.validation.detection_map50}% mAP@0.5</dd></div>
                <div><dt>Tracking</dt><dd>{contest.mot17.mota} MOTA</dd></div>
                <div><dt>Pipeline</dt><dd>≈{contest.footprint.pipeline_fps_approx} FPS</dd></div>
              </dl>
            </article>
            <article className="evidence-card">
              <span className="tag tag--bronze">Submitted poster snapshot</span>
              <h3>What the judges saw</h3>
              <div className="evidence-number">{results.contest_submission_snapshot.reported_fps} FPS</div>
              <p>Archived presentation value, preserved as submitted rather than silently rewritten.</p>
              <div className="reconciliation">
                <ArrowDownRight />
                <span>Canonical reconciliation</span>
                <strong>7.78M · ≈18 full-pipeline FPS</strong>
              </div>
            </article>
            <article className="evidence-card">
              <span className="tag tag--muted">Responsible reporting</span>
              <h3>No protocol blending</h3>
              <p>
                The poster’s +16.2 point row combined several changes. The matched post-contest
                part-readout study reports <strong>+{String(partjde.matched_part_readout_gain_pp)} pp</strong>.
              </p>
              <a href="/report/effiped-technical-report.pdf">Read the evidence notes <BookOpen size={16} /></a>
            </article>
          </div>

          <div className="protocol-table" role="table" aria-label="Contest metric provenance">
            <div className="protocol-table__head" role="row">
              <span role="columnheader">Evaluation</span><span role="columnheader">Rank-1</span><span role="columnheader">Detection</span><span role="columnheader">Boundary</span>
            </div>
            <div role="row">
              <span role="cell">P-DESTRE validation</span><strong role="cell">{contest.pdestre.validation.rank1_cross}%</strong><span role="cell">{contest.pdestre.validation.detection_map50}% mAP@0.5</span><small role="cell">Protocol D</small>
            </div>
            <div role="row">
              <span role="cell">P-DESTRE test</span><strong role="cell">{contest.pdestre.test.rank1_cross}%</strong><span role="cell">{contest.pdestre.test.detection_map50}% mAP@0.5</span><small role="cell">Protocol E</small>
            </div>
            <div role="row">
              <span role="cell">MOT17 val-half</span><strong role="cell">{contest.mot17.idf1} IDF1</strong><span role="cell">{contest.mot17.hota} HOTA</span><small role="cell">Protocol A</small>
            </div>
          </div>
        </section>

        <section className="section evolution">
          <div className="section__intro">
            <span className="section-index">05</span>
            <div>
              <p className="eyebrow">Research evolution</p>
              <h2>The contest prototype became a controlled research question.</h2>
            </div>
          </div>
          <div className="evolution__rail">
            <article>
              <span>01 · Contest</span>
              <Award />
              <h3>EffiPed</h3>
              <p>Joint detection, part-based descriptors, tracking, and four-camera identity review.</p>
              <strong>3rd Prize · SIPC 2026</strong>
            </article>
            <article>
              <span>02 · Matched refinement</span>
              <Layers3 />
              <h3>PartJDE</h3>
              <p>Isolated the value of part-based RoI-strip readout under a tighter comparison.</p>
              <strong>+{String(partjde.matched_part_readout_gain_pp)} pp Rank-1</strong>
            </article>
            <article>
              <span>03 · Five-fold evidence</span>
              <Box />
              <h3>BoxJDE</h3>
              <p>Reduced the claim to full-box descriptor support inside a matched efficient JDE model.</p>
              <strong>+{String(boxjde.natural_predicted_rank1_gain_pp)} pp predicted-box R1</strong>
              <a href={boxjdeRepository} target="_blank" rel="noreferrer">Open BoxJDE <ExternalLink size={15} /></a>
            </article>
          </div>
          <p className="protocol-warning">
            <ShieldAlert size={19} />
            BoxJDE’s primary P-DESTRE study is a constructed per-date readout ablation, not the official Task 4 benchmark.
          </p>
        </section>

        <section className="section engineering">
          <div className="section__intro">
            <span className="section-index">06</span>
            <div>
              <p className="eyebrow">Engineering depth</p>
              <h2>A research model, a review product, and a reproducible evidence trail.</h2>
            </div>
          </div>
          <div className="engineering__grid">
            {[
              ["Model", "ConvNeXt V2 · CenterNet · RoIAlign · CoordinateAttention"],
              ["Learning", "Sub-center ArcFace · triplet loss · XBM · BNNeck"],
              ["Tracking", "BoT-SORT · Kalman motion · cosine gallery association"],
              ["Product", "FastAPI · React · WebSocket progress · explicit cleanup"],
              ["Evidence", "P-DESTRE validation/test · MOT17 val-half · checked fixtures"],
              ["Delivery", "Python package · Docker-ready runtime · Vercel static replay"]
            ].map(([title, copy]) => (
              <article key={title}><CheckCircle2 /><span>{title}</span><p>{copy}</p></article>
            ))}
          </div>
        </section>

        <section className="section local-run">
          <div>
            <p className="eyebrow">Run the live workflow locally</p>
            <h2>Vercel hosts the replay. Your GPU runs the model.</h2>
            <p>
              Public weights remain withheld while dataset redistribution terms are unresolved.
              Supply an authorized checkpoint in <code>EFFIPED_WEIGHTS_DIR</code>.
            </p>
          </div>
          <pre><code>{`python -m venv .venv
pip install -e ".[runtime]"
effiped-app

# then open http://127.0.0.1:8000`}</code></pre>
        </section>
      </main>

      <footer>
        <div className="brand"><span className="brand__mark">E</span><span>EffiPed</span></div>
        <p>
          Software © 2026 Aswanth Raj · Apache-2.0. P-DESTRE-derived media is separately licensed
          CC BY-NC-SA 4.0 for this non-commercial research showcase.
        </p>
        <div>
          <a href={repository}>GitHub</a>
          <a href="/media/ASSET_MANIFEST.json">Media manifest</a>
          <a href="https://arxiv.org/abs/2004.02782">P-DESTRE paper</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
