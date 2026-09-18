import { ArrowRight, Boxes, Camera, Network, ScanSearch } from "lucide-react";

import { Callout, Label, Metric, Section, Step } from "../components/page";
import { Link } from "../lib/router";
import { boxjdeRepository, repository } from "../lib/site";
import { results } from "../data/results";
import { personSearch, sessionStats } from "../data/personSearch";

export function Overview() {
  const benchmark = results.system_benchmarks;
  const session = results.demo_case.session_diagnostic;

  return (
    <>
      <header className="hero">
        <div className="hero__copy">
          <Label>Pedestrian re-identification</Label>
          <h1>Multi-Camera Pedestrian Identity Review</h1>
          <p className="hero__formal">{results.project.title}</p>
          <p className="hero__lede">
            EffiPed detects pedestrians across several fixed cameras and ranks which observations
            plausibly show the same person, so that an analyst can review them. One compact network
            does the detection and the identity descriptor together, which is what lets the whole
            pipeline run on a single GPU.
          </p>
          <p className="hero__proof">
            <strong>{benchmark.pdestre.validation.rank1_cross}%</strong> cross-camera Rank-1 on
            P-DESTRE fold-0 validation, at{" "}
            <strong>{benchmark.footprint.parameters_m}M</strong> parameters and roughly{" "}
            <strong>{benchmark.footprint.pipeline_fps_approx} FPS</strong> for the complete pipeline.
          </p>
          <div className="hero__actions">
            <Link className="primary-link" to="/workbench">
              Open the workbench <ArrowRight aria-hidden="true" size={16} />
            </Link>
            <Link className="ghost-link" to="/evidence">
              See the evidence
            </Link>
          </div>
        </div>

        <figure className="hero__media">
          <video
            aria-label="Four synchronized camera views with detector and tracker annotations"
            autoPlay
            loop
            muted
            playsInline
            poster="/media/pdestre/four-camera-tracking.jpg"
            preload="metadata"
            src="/media/pdestre/multi-camera-tracking.webm"
          />
          <figcaption>
            <span className="live-dot" aria-hidden="true" />
            Four synchronized views from P-DESTRE session 12-11-2019_3. The boxes were drawn by the
            tracking pipeline, not by the browser.
          </figcaption>
        </figure>
      </header>

      <Section
        id="figures"
        index="01"
        label="Measured"
        title="Four protocols, reported separately."
        lede="Detection, tracking, cross-camera retrieval and efficiency answer different questions, so they are never averaged into one score. Each figure below carries what it was measured on."
      >
        <div className="metric-row">
          <Metric
            name="Cross-camera Rank-1"
            protocol="P-DESTRE fold-0 validation, Protocol D"
            tone="link"
            unit="%"
            value={String(benchmark.pdestre.validation.rank1_cross)}
          />
          <Metric
            name="Held-out Rank-1"
            protocol="P-DESTRE fold-0 test, Protocol E"
            tone="plain"
            unit="%"
            value={String(benchmark.pdestre.test.rank1_cross)}
          />
          <Metric
            name="Detection mAP@0.5"
            protocol="P-DESTRE fold-0 validation"
            tone="plain"
            unit="%"
            value={String(benchmark.pdestre.validation.detection_map50)}
          />
          <Metric
            name="MOT17 MOTA"
            protocol="Val-half Protocol A, held out of training"
            tone="plain"
            value={benchmark.mot17.mota.toFixed(2)}
          />
          <Metric
            name="Parameters"
            protocol={`About ${benchmark.footprint.pipeline_fps_approx} FPS end to end`}
            tone="note"
            unit="M"
            value={String(benchmark.footprint.parameters_m)}
          />
        </div>
        <p className="section__foot">
          Every value is read from a single checked-in fixture and pinned by a validator in CI, so a
          published number cannot drift without the check failing.{" "}
          <Link className="text-link-inline" to="/evidence">
            The full ledger, with what each figure does not establish
          </Link>
        </p>
      </Section>

      <Section
        index="02"
        label="What it does"
        title="Detect, hold identity inside a camera, then compare across cameras."
        lede="Motion keeps an identity together inside one view. It cannot bridge two views, so the third step has only appearance to work with, and appearance alone is never certainty."
      >
        <div className="steps">
          <Step n="01" title="Detect at stride 4">
            A ConvNeXt V2 backbone with P2 and P3 fusion feeds CenterNet outputs. Working at stride 4
            keeps small and distant people in the output, which aerial footage is full of.
          </Step>
          <Step n="02" title="Read identity from body parts">
            RoIAlign crops each person to 32 by 8, four horizontal strips keep local appearance, and
            Coordinate Attention fuses the visible ones into a{" "}
            {benchmark.footprint.descriptor_dim}-D descriptor. A diversity term during training stops
            the four strips collapsing onto the same feature.
          </Step>
          <Step n="03" title="Track inside each camera">
            BoT-SORT combines motion, overlap and appearance so a person keeps one track identity
            while they stay in view, through short occlusions and scale changes.
          </Step>
          <Step n="04" title="Rank candidates across cameras">
            Descriptors from other views are ranked by cosine similarity and presented as ordered
            candidate evidence. A person decides; the system never does.
          </Step>
        </div>
      </Section>

      <Section
        index="03"
        label="The application"
        title="A six-panel review workbench, running on four real camera clips."
        lede="The workbench is the product. It mirrors the layout of the desktop application the archived results came from, with four P-DESTRE clips already attached as though you had uploaded them."
      >
        <div className="panel-grid">
          {[
            ["Single Camera", "Per-camera tracked output and a run summary", Camera],
            ["Cross Camera", "Four-view association replay and per-pair precision", Network],
            ["Person Search", "Build the index, pick anyone, review ranked candidates", ScanSearch],
            ["Image Detection", "A single-frame detection pass", Boxes]
          ].map(([name, copy, Icon]) => {
            const PanelIcon = Icon as typeof Camera;
            return (
              <article className="panel-card" key={name as string}>
                <PanelIcon aria-hidden="true" size={17} />
                <strong>{name as string}</strong>
                <p>{copy as string}</p>
              </article>
            );
          })}
        </div>

        <div className="session-strip">
          <div>
            <span>{session.cameras}</span>
            <small>cameras</small>
          </div>
          <div>
            <span>{session.frames}</span>
            <small>frames</small>
          </div>
          <div>
            <span>{session.local_tracks}</span>
            <small>local tracks</small>
          </div>
          <div>
            <span>{personSearch.people.length}</span>
            <small>indexed people</small>
          </div>
          <div>
            <span>{session.cross_camera_ids}</span>
            <small>cross-camera identities</small>
          </div>
          <div>
            <span>{Math.round(sessionStats.precision * 100)}%</span>
            <small>pairwise precision</small>
          </div>
        </div>
        <p className="section__foot">
          {session.label}. Those six numbers describe this one archived session and are not a
          benchmark result.
        </p>

        <Link className="primary-link" to="/workbench">
          Open the workbench <ArrowRight aria-hidden="true" size={16} />
        </Link>
      </Section>

      <Section
        index="04"
        label="Boundaries"
        title="What this is, and what it is not."
      >
        <Callout title="A ranked match is not an identification" tone="warning">
          <p>
            Cross-camera similarity orders candidate appearance evidence for a person to review. The
            demo data shows exactly why the ranking matters more than the number: non-matches score
            between 0.997 and 0.998 against matches at 0.998 to 0.999. There is no calibration and no
            decision threshold. This system does not do face recognition or biometric identification,
            and it makes no automated identity decision.
          </p>
        </Callout>
        <Callout title="The weights are withheld, so this page cannot infer" tone="note">
          <p>
            Redistribution terms across the four training sources are unresolved, so no checkpoint is
            published. The hosted build performs no inference: the controls are live, and pressing
            Run replays an archived result. The same interface connects to local GPU inference when
            you supply an authorized checkpoint.
          </p>
        </Callout>
      </Section>

      <Section index="05" label="Continue" title="Where to go next.">
        <div className="next-rows">
          <Link className="next-row" to="/system">
            <span className="next-row__n">01</span>
            <div>
              <strong>How the system works</strong>
              <p>One backbone, two heads, two scales of association, and why each choice was made.</p>
            </div>
            <ArrowRight aria-hidden="true" size={16} />
          </Link>
          <Link className="next-row" to="/evidence">
            <span className="next-row__n">02</span>
            <div>
              <strong>Evidence and protocols</strong>
              <p>
                Every published figure with its protocol, the descriptor research that followed, and
                the limits of each claim.
              </p>
            </div>
            <ArrowRight aria-hidden="true" size={16} />
          </Link>
          <Link className="next-row" to="/deploy">
            <span className="next-row__n">03</span>
            <div>
              <strong>Run it locally</strong>
              <p>Install the package, start the service, and what the withheld weights mean for you.</p>
            </div>
            <ArrowRight aria-hidden="true" size={16} />
          </Link>
          <a className="next-row" href={repository} rel="noreferrer" target="_blank">
            <span className="next-row__n">04</span>
            <div>
              <strong>Read the source</strong>
              <p>
                The model, the service, the validators, and the licence audit that decided what could
                be published. Follow-on descriptor research lives in{" "}
                <span className="next-row__aside">{boxjdeRepository.replace("https://github.com/", "")}</span>.
              </p>
            </div>
            <ArrowRight aria-hidden="true" size={16} />
          </a>
        </div>
      </Section>
    </>
  );
}
