import { Video } from "lucide-react";

import type { RunResult } from "../engine/pipeline";

export function SourcesView({
  result,
  onOpenSources
}: {
  result: RunResult | null;
  onOpenSources: () => void;
}) {
  if (!result || result.sources.length === 0) {
    return (
      <div className="empty">
        <Video aria-hidden="true" size={22} />
        <p>No clips attached.</p>
        <button className="primary-button" onClick={onOpenSources} type="button">
          Select video sources
        </button>
      </div>
    );
  }

  return (
    <div className="view view--sources">
      <div className="view__head">
        <h2>Attached sources</h2>
        <button className="ghost-button" onClick={onOpenSources} type="button">
          Change selection
        </button>
      </div>

      <div className="clip-grid">
        {result.sources.map((video) => {
          const rows = result.people.filter((row) => row.person.video_index === video.id);
          const detections = rows.reduce((total, row) => total + row.detections.length, 0);
          return (
            <figure className="clip" key={video.id}>
              <video
                aria-label={`${video.label} source footage`}
                controls
                loop
                muted
                playsInline
                poster={video.poster}
                preload="metadata"
                src={video.source}
              />
              <figcaption>
                <div className="clip__title">
                  <strong>{video.label}</strong>
                  <code>{video.file_name}</code>
                </div>
                <dl className="kv">
                  <div>
                    <dt>Resolution</dt>
                    <dd>{video.width} x {video.height}</dd>
                  </div>
                  <div>
                    <dt>Frames</dt>
                    <dd>{video.frame_count} at {video.fps} fps</dd>
                  </div>
                  <div>
                    <dt>Tracks kept</dt>
                    <dd>{rows.length}</dd>
                  </div>
                  <div>
                    <dt>Detections</dt>
                    <dd>{detections}</dd>
                  </div>
                </dl>
              </figcaption>
            </figure>
          );
        })}
      </div>

      <p className="view__foot">
        P-DESTRE session 12-11-2019_3, adapted under CC BY-NC-SA 4.0. The boxes burned into the
        tracked renders were drawn by the original pipeline, not by this interface.
      </p>
    </div>
  );
}
