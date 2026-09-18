import { useEffect, useRef, useState } from "react";
import { Check, X } from "lucide-react";

import { personSearch } from "../data/personSearch";

/**
 * A native <dialog>, so the top layer, the backdrop, the focus trap and Escape
 * come from the platform rather than from hand-rolled key handling that has to
 * be right on every browser.
 */
export function SourceModal({
  open,
  selected,
  onCancel,
  onConfirm
}: {
  open: boolean;
  selected: number[];
  onCancel: () => void;
  onConfirm: (ids: number[]) => void;
}) {
  const [draft, setDraft] = useState<number[]>(selected);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open) {
      setDraft(selected);
      if (!element.open) element.showModal();
    } else if (element.open) {
      element.close();
    }
  }, [open, selected]);

  const toggle = (id: number) =>
    setDraft((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id].sort()
    );

  return (
    <dialog
      aria-labelledby="source-modal-title"
      className="modal"
      onCancel={(event) => {
        event.preventDefault();
        onCancel();
      }}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog element itself.
        if (event.target === dialog.current) onCancel();
      }}
      ref={dialog}
    >
      <div className="modal__inner">
        <header className="modal__head">
          <h2 id="source-modal-title">Select video sources</h2>
          <button aria-label="Close" className="icon-button" onClick={onCancel} type="button">
            <X aria-hidden="true" size={16} />
          </button>
        </header>

        <p className="modal__note">
          Four synchronized clips from P-DESTRE session 12-11-2019_3 are attached to this job. Pick
          the ones the pipeline should process.
        </p>

        <div className="source-grid">
          {personSearch.videos.map((video) => {
            const checked = draft.includes(video.id);
            const tracks = personSearch.people.filter((p) => p.video_index === video.id).length;
            return (
              <button
                aria-pressed={checked}
                className={checked ? "source-card is-on" : "source-card"}
                key={video.id}
                onClick={() => toggle(video.id)}
                type="button"
              >
                <span className="source-card__frame">
                  <img alt="" className="source-card__poster" loading="lazy" src={video.poster} />
                  <span className="source-card__check" aria-hidden="true">
                    {checked ? <Check size={11} strokeWidth={3} /> : null}
                  </span>
                </span>
                <span className="source-card__body">
                  <span className="source-card__title">
                    <strong>{video.label}</strong>
                    <code>{video.file_name}</code>
                  </span>
                  <dl className="source-card__spec">
                    <div>
                      <dt>Frame</dt>
                      <dd>{video.width}x{video.height}</dd>
                    </div>
                    <div>
                      <dt>Length</dt>
                      <dd>{video.duration_s.toFixed(0)}s at {video.fps}fps</dd>
                    </div>
                    <div>
                      <dt>Frames</dt>
                      <dd>{video.frame_count}</dd>
                    </div>
                    <div>
                      <dt>Tracks</dt>
                      <dd className="is-accent">{tracks}</dd>
                    </div>
                  </dl>
                </span>
              </button>
            );
          })}
        </div>

        <footer className="modal__foot">
          <div className="modal__bulk">
            <button className="ghost-button" onClick={() => setDraft([0, 1, 2, 3])} type="button">
              Select all
            </button>
            <button className="ghost-button" disabled={draft.length === 0} onClick={() => setDraft([])} type="button">
              Clear
            </button>
          </div>
          <div className="modal__actions">
            <span className="modal__count">
              {draft.length} of {personSearch.videos.length} selected
            </span>
            <button className="ghost-button" onClick={onCancel} type="button">
              Cancel
            </button>
            <button
              className="run-button run-button--compact"
              disabled={draft.length === 0}
              onClick={() => onConfirm(draft)}
              type="button"
            >
              Attach {draft.length === 1 ? "1 clip" : `${draft.length} clips`}
            </button>
          </div>
        </footer>
      </div>
    </dialog>
  );
}
