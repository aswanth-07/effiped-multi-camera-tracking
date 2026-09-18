import { useEffect, useRef, useState } from "react";
import { Check, Film, X } from "lucide-react";

import { personSearch } from "../data/personSearch";

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
  const dialog = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) setDraft(selected);
  }, [open, selected]);

  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onCancel();
        return;
      }
      if (event.key !== "Tab") return;
      // A modal that lets focus escape into the page behind it is not a modal.
      const focusable = dialog.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  const toggle = (id: number) =>
    setDraft((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id].sort()
    );

  return (
    <div className="modal-scrim" role="presentation">
      <div
        aria-labelledby="source-modal-title"
        aria-modal="true"
        className="modal"
        ref={dialog}
        role="dialog"
      >
        <header className="modal__head">
          <div>
            <p className="modal__eyebrow">Job input</p>
            <h2 id="source-modal-title">Select video sources</h2>
          </div>
          <button aria-label="Close" className="icon-button" onClick={onCancel} ref={closeButton} type="button">
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
            const people = personSearch.people.filter((person) => person.video_index === video.id).length;
            return (
              <button
                aria-pressed={checked}
                className={checked ? "source-card is-on" : "source-card"}
                key={video.id}
                onClick={() => toggle(video.id)}
                type="button"
              >
                <span className="source-card__check" aria-hidden="true">
                  {checked ? <Check size={12} /> : null}
                </span>
                <img alt="" className="source-card__poster" loading="lazy" src={video.poster} />
                <span className="source-card__body">
                  <strong>{video.label}</strong>
                  <code>{video.file_name}</code>
                  <span className="source-card__meta">
                    <span>{video.width} x {video.height}</span>
                    <span>{video.duration_s.toFixed(0)}s</span>
                    <span>{video.fps} fps</span>
                    <span>{video.frame_count} frames</span>
                  </span>
                  <span className="source-card__tracks">
                    <Film aria-hidden="true" size={11} /> {people} tracks indexed
                  </span>
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
            <button className="ghost-button" onClick={() => setDraft([])} type="button">
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
              className="primary-button"
              disabled={draft.length === 0}
              onClick={() => onConfirm(draft)}
              type="button"
            >
              Attach {draft.length === 1 ? "1 clip" : `${draft.length} clips`}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
