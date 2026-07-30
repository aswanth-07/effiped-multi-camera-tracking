import { useMemo, useState } from "react";
import { Camera, Crosshair, LocateFixed, Play, ShieldCheck } from "lucide-react";
import { results } from "../data/results";

export function Investigation() {
  const demo = results.demo_case;
  const [activeCamera, setActiveCamera] = useState(demo.cameras[0].id);
  const [selectedSubject, setSelectedSubject] = useState(demo.subjects[0].id);
  const subject = useMemo(
    () => demo.subjects.find((item) => item.id === selectedSubject) ?? demo.subjects[0],
    [demo.subjects, selectedSubject]
  );
  const camera = demo.cameras.find((item) => item.id === activeCamera) ?? demo.cameras[0];
  const visible = demo.subjects
    .flatMap((item) => item.appearances.map((appearance) => ({ ...appearance, subject: item })))
    .filter((appearance) => appearance.camera === camera.id);
  const candidates = subject.appearances
    .filter((appearance) => appearance.camera !== activeCamera)
    .sort((a, b) => b.similarity - a.similarity);

  return (
    <div className="investigation">
      <header className="investigation__top">
        <div>
          <span className="label">Precomputed identity review</span>
          <h3>{demo.title}</h3>
        </div>
        <div className="demo-pill"><Play size={14} /> Research replay · no upload</div>
      </header>

      <div className="investigation__workspace">
        <div className="camera-column">
          <div className="camera-tabs" role="tablist" aria-label="Camera source">
            {demo.cameras.map((item) => (
              <button
                role="tab"
                aria-selected={item.id === activeCamera}
                className={item.id === activeCamera ? "active" : ""}
                onClick={() => setActiveCamera(item.id)}
                key={item.id}
              >
                <Camera size={15} />
                {item.id}
              </button>
            ))}
          </div>
          <div className={`camera-view camera-view--${camera.position}`}>
            <div className="camera-view__hud">
              <span>REC · {camera.time}</span>
              <span>{camera.label}</span>
            </div>
            {visible.map((appearance) => (
              <button
                aria-label={`Select ${appearance.subject.id}`}
                className={`track-box ${selectedSubject === appearance.subject.id ? "selected" : ""}`}
                key={`${appearance.subject.id}-${appearance.camera}`}
                onClick={() => setSelectedSubject(appearance.subject.id)}
                style={{
                  left: `${appearance.bbox[0]}%`,
                  top: `${appearance.bbox[1]}%`,
                  width: `${appearance.bbox[2]}%`,
                  height: `${appearance.bbox[3]}%`,
                  borderColor: appearance.subject.color,
                  color: appearance.subject.color
                }}
              >
                <span>{appearance.subject.id}</span>
              </button>
            ))}
            <div className="scan-line" />
          </div>
          <div className="timeline" aria-label="Camera timeline">
            <span>00:00</span>
            <div><i style={{ left: "39%" }} /></div>
            <span>00:15</span>
          </div>
        </div>

        <aside className="candidate-panel">
          <div className="candidate-panel__query">
            <Crosshair />
            <div>
              <span>Query under review</span>
              <strong>{subject.id}</strong>
              <small>{subject.label}</small>
            </div>
          </div>
          <div className="candidate-panel__heading">
            <span>Cross-camera candidates</span>
            <ShieldCheck size={16} />
          </div>
          <div className="candidate-list">
            {candidates.map((candidate, index) => {
              const target = demo.cameras.find((item) => item.id === candidate.camera)!;
              const tone = candidate.similarity >= 0.8 ? "strong" : candidate.similarity >= 0.5 ? "possible" : "low";
              return (
                <button
                  className={`candidate candidate--${tone}`}
                  key={candidate.camera}
                  onClick={() => setActiveCamera(candidate.camera)}
                >
                  <span className={`candidate__thumb camera-view--${target.position}`} />
                  <span className="candidate__copy">
                    <small>#{index + 1} · {target.label}</small>
                    <strong>{Math.round(candidate.similarity * 100)}% similarity</strong>
                    <span>{tone === "strong" ? "Strong candidate" : tone === "possible" ? "Possible candidate" : "Low confidence"}</span>
                  </span>
                  <LocateFixed size={17} />
                </button>
              );
            })}
          </div>
          <p className="candidate-panel__note">
            Similarity ranks evidence for review. It does not establish a person’s identity.
          </p>
        </aside>
      </div>
    </div>
  );
}

