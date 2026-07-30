import { ArrowRight, Boxes, BrainCircuit, ScanSearch, Share2 } from "lucide-react";

const blocks = [
  { kicker: "01 · perception", title: "ConvNeXt V2", detail: "P2 stride-4 + P3 stride-8", icon: BrainCircuit },
  { kicker: "02 · fusion", title: "Adaptive P2/P3", detail: "Fine edges + semantic context", icon: Share2 },
  { kicker: "03 · detection", title: "CenterNet", detail: "Heatmap · box · offset · IoU", icon: ScanSearch },
  { kicker: "04 · identity", title: "Part ReID", detail: "RoIAlign · 4 strips · CoordAttn", icon: Boxes }
];

export function Architecture() {
  return (
    <div className="architecture" aria-label="EffiPed model architecture">
      <div className="architecture__rail">
        {blocks.map((block, index) => {
          const Icon = block.icon;
          return (
            <div className="architecture__step-wrap" key={block.title}>
              <article className="architecture__step">
                <span>{block.kicker}</span>
                <Icon aria-hidden="true" />
                <strong>{block.title}</strong>
                <small>{block.detail}</small>
              </article>
              {index < blocks.length - 1 && <ArrowRight className="architecture__arrow" aria-hidden="true" />}
            </div>
          );
        })}
      </div>
      <div className="architecture__branch">
        <div>
          <span>Temporal path</span>
          <strong>BoT-SORT + Kalman motion</strong>
          <code>track<sub>t</sub> ← IoU + cosine(z<sub>t</sub>, z̄)</code>
        </div>
        <div>
          <span>Cross-camera path</span>
          <strong>Gallery candidate ranking</strong>
          <code>s(q,g) = Σ a<sub>k</sub> cos(z<sub>q,k</sub>, z<sub>g,k</sub>)</code>
        </div>
        <div className="architecture__output">
          <span>Review output</span>
          <strong>{"{box, local track, 256-D identity}"}</strong>
          <small>Candidate evidence for human review — not proof of identity</small>
        </div>
      </div>
    </div>
  );
}
