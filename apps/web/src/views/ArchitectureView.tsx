import { Maximize2 } from "lucide-react";

import {
  CONFIG,
  DETECTION_LOSSES,
  DIAGRAM,
  FOOTPRINT,
  IDENTITY_LOSSES,
  INFERENCE,
  RESULTS,
  TRAINING,
  type Fact,
  type LossTerm
} from "../data/architecture";

function LossTable({ caption, sub, terms }: { caption: string; sub: string; terms: LossTerm[] }) {
  return (
    <div className="arch-loss">
      <table className="grid-table">
        <caption>
          {caption}, L<sub>{sub}</sub>
        </caption>
        <thead>
          <tr>
            <th scope="col">Term</th>
            <th scope="col">Loss</th>
            <th className="is-num" scope="col">
              Weight
            </th>
          </tr>
        </thead>
        <tbody>
          {terms.map((term) => (
            <tr key={term.name}>
              <th scope="row">{term.name}</th>
              <td>{term.loss}</td>
              <td className="is-num">{term.weight}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Facts({ id, title, facts }: { id: string; title: string; facts: Fact[] }) {
  return (
    <section aria-labelledby={id} className="arch-facts">
      <h3 id={id}>{title}</h3>
      <dl>
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export function ArchitectureView() {
  return (
    <div className="view view--architecture">
      <div className="view__head">
        <h2>Architecture</h2>
        <span className="arch-spec">
          {FOOTPRINT.parameters_m}M parameters · {FOOTPRINT.input_resolution} · {CONFIG["model.embedding_dim"]}-D
          descriptor · one network for detection and identity
        </span>
      </div>

      <div className="arch-layout">
        <figure className="arch-figure">
          {/* On a phone the figure keeps a readable size and scrolls sideways, so the
              scroller has to be reachable from the keyboard as well. */}
          <div aria-label="Architecture diagram" className="arch-figure__scroll" role="region" tabIndex={0}>
            <img alt={DIAGRAM.alt} height={DIAGRAM.height} src={DIAGRAM.src} width={DIAGRAM.width} />
          </div>
          <figcaption>
            <a href={DIAGRAM.src} rel="noreferrer" target="_blank">
              <Maximize2 aria-hidden="true" size={11} />
              Full size
            </a>
          </figcaption>
        </figure>

        <div className="arch-side">
          <section aria-labelledby="arch-objective" className="arch-objective">
            <h3 id="arch-objective">Training objective</h3>
            <p className="arch-formula">
              L = ½ ( e<sup>−s<sub>det</sub></sup> L<sub>det</sub> + s<sub>det</sub> + e<sup>−s<sub>id</sub></sup>{" "}
              L<sub>id</sub> + s<sub>id</sub> )
            </p>
            <p className="arch-formula__key">
              s<sub>det</sub>, s<sub>id</sub>: learned task uncertainties, initialised at −1.85 and −1.05
            </p>
            <LossTable caption="Detection" sub="det" terms={DETECTION_LOSSES} />
            <LossTable caption="Identity" sub="id" terms={IDENTITY_LOSSES} />
          </section>
          <Facts facts={RESULTS} id="arch-results" title="Reported results" />
        </div>
      </div>

      <div className="arch-columns">
        <Facts facts={TRAINING} id="arch-training" title="Training" />
        <Facts facts={INFERENCE} id="arch-inference" title="Inference" />
      </div>
    </div>
  );
}
