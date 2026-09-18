import { AlertTriangle, ExternalLink } from "lucide-react";

import { results } from "../data/results";
import { personSearch } from "../data/personSearch";
import { INDEX_SETTINGS } from "../engine/pipeline";
import { boxjdeRepository, repository } from "../lib/site";

export function ModelView() {
  const bench = results.system_benchmarks;
  const fp = bench.footprint;

  return (
    <div className="view view--model">
      <div className="view__head">
        <h2>Model and runtime</h2>
        <a className="ghost-button" href={repository} rel="noreferrer" target="_blank">
          Repository <ExternalLink aria-hidden="true" size={12} />
        </a>
      </div>

      <div className="panels">
        <section className="panel">
          <p className="panel-label">Loaded artifact</p>
          <dl className="kv kv--wide">
            <div><dt>Model</dt><dd>{bench.model}</dd></div>
            <div><dt>Backbone</dt><dd>ConvNeXt V2 Tiny, stride-4 P2 fusion</dd></div>
            <div><dt>Heads</dt><dd>CenterNet detection, four-strip part ReID</dd></div>
            <div><dt>Descriptor</dt><dd>{fp.descriptor_dim}-D, BNNeck at eval</dd></div>
            <div><dt>Parameters</dt><dd>{fp.parameters_m}M</dd></div>
            <div><dt>Input</dt><dd>{fp.input_resolution}</dd></div>
            <div><dt>Throughput</dt><dd>about {fp.pipeline_fps_approx} FPS end to end</dd></div>
            <div><dt>Device</dt><dd>{fp.device}</dd></div>
            <div><dt>Weights</dt><dd className="is-warn">withheld, dataset rights review</dd></div>
          </dl>
        </section>

        <section className="panel">
          <p className="panel-label">This replay</p>
          <dl className="kv kv--wide">
            <div><dt>Job</dt><dd><code>{personSearch.job.job_id}</code></dd></div>
            <div><dt>Index built with</dt><dd>{personSearch.generated_with.checkpoint}</dd></div>
            <div><dt>Indexed people</dt><dd>{personSearch.people.length}</dd></div>
            <div><dt>Frame sets</dt><dd>{personSearch.job.total_frame_sets}</dd></div>
            <div><dt>Decode floor</dt><dd>{INDEX_SETTINGS.decode_thresh}</dd></div>
            <div><dt>Track floor</dt><dd>{INDEX_SETTINGS.track_thresh}</dd></div>
            <div><dt>Frame stride</dt><dd>{INDEX_SETTINGS.frame_stride}</dd></div>
            <div><dt>Views per person</dt><dd>{INDEX_SETTINGS.max_views_per_person}</dd></div>
          </dl>
          <p className="panel__note">
            <AlertTriangle aria-hidden="true" size={13} />
            <span>
              The index a visitor browses was computed offline with the BoxJDE research checkpoint,
              because the Tier-1 weights are withheld. It is therefore not the output of the model
              the benchmarks below describe.
            </span>
          </p>
        </section>

        <section className="panel panel--wide">
          <p className="panel-label">Benchmarks, by protocol</p>
          <table className="grid-table">
            <thead>
              <tr>
                <th scope="col">Evaluation</th>
                <th scope="col">Metric</th>
                <th className="is-num" scope="col">Value</th>
              </tr>
            </thead>
            <tbody>
              <tr className="is-good"><th scope="row">P-DESTRE fold-0 validation</th><td>Cross-camera Rank-1</td><td className="is-num">{bench.pdestre.validation.rank1_cross.toFixed(2)}</td></tr>
              <tr><th scope="row">P-DESTRE fold-0 validation</th><td>Detection mAP@0.5</td><td className="is-num">{bench.pdestre.validation.detection_map50.toFixed(2)}</td></tr>
              <tr><th scope="row">P-DESTRE fold-0 test</th><td>Cross-camera Rank-1</td><td className="is-num">{bench.pdestre.test.rank1_cross.toFixed(2)}</td></tr>
              <tr><th scope="row">P-DESTRE fold-0 test</th><td>Detection mAP@0.5</td><td className="is-num">{bench.pdestre.test.detection_map50.toFixed(2)}</td></tr>
              <tr><th scope="row">MOT17 val-half</th><td>MOTA</td><td className="is-num">{bench.mot17.mota.toFixed(2)}</td></tr>
              <tr><th scope="row">MOT17 val-half</th><td>IDF1</td><td className="is-num">{bench.mot17.idf1.toFixed(2)}</td></tr>
              <tr><th scope="row">MOT17 val-half</th><td>HOTA</td><td className="is-num">{bench.mot17.hota.toFixed(2)}</td></tr>
            </tbody>
          </table>
          <p className="panel__note">
            <span>
              Every value is read from one checked-in fixture that a validator pins, so a published
              number cannot drift silently. The browser replay produces none of them. Follow-on
              descriptor research lives in the{" "}
              <a href={boxjdeRepository} rel="noreferrer" target="_blank">BoxJDE repository</a> and
              uses a constructed five-fold protocol, not official P-DESTRE Task 4.
            </span>
          </p>
        </section>

        <section className="panel panel--wide">
          <p className="panel-label">Boundaries</p>
          <ul className="bullets">
            <li><b>A ranked match is not an identification.</b> Scores order candidate evidence for a person to review, and are uncalibrated.</li>
            <li><b>No inference runs in this browser.</b> Thresholds filter stored detections and similarities; they cannot create a detection the model never made.</li>
            <li><b>One fold, one dataset, single measurements.</b> No variance, interval or seed policy is published for any figure.</li>
            <li><b>No fairness or subgroup analysis exists</b>, on a system that ranks people by appearance.</li>
            <li><b>No face recognition and no biometric identification.</b> Deployment would need its own privacy, consent, retention and bias assessment.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
