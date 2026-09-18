import { Search } from "lucide-react";

import type { RunResult, Settings } from "../engine/pipeline";

function confidenceClass(score: number): string {
  if (score >= 0.66) return "is-high";
  if (score >= 0.6) return "is-mid";
  return "is-low";
}

export function SearchView({
  result,
  settings,
  selected,
  onSelect
}: {
  result: RunResult | null;
  settings: Settings;
  selected: string | null;
  onSelect: (personId: string) => void;
}) {
  if (!result) {
    return <div className="empty"><Search aria-hidden="true" size={22} /><p>Run the pipeline to build the person index.</p></div>;
  }

  const row = selected ? result.byId[selected] : null;

  return (
    <div className="view view--search">
      <div className="view__head">
        <h2>Person search index</h2>
        <span className="view__count">
          {result.people.length} indexed across {result.sources.length} clips
        </span>
      </div>

      {result.people.length === 0 ? (
        <div className="empty"><p>No track survives the current thresholds. Lower the activation or confidence floor.</p></div>
      ) : (
        <div className="search-index">
        <div className="gallery" role="listbox" aria-label="Indexed people">
          {result.people.map((entry) => {
            const isOn = entry.person.id === selected;
            return (
              <button
                aria-selected={isOn}
                className={isOn ? "tile is-on" : "tile"}
                key={entry.person.id}
                onClick={() => onSelect(entry.person.id)}
                role="option"
                type="button"
              >
                {entry.person.crop ? (
                  <img alt="" className="tile__crop" loading="lazy" src={entry.person.crop} />
                ) : (
                  <span className="tile__crop tile__crop--none" />
                )}
                <span className="tile__meta">
                  <code>v{entry.person.video_index + 1}/t{entry.person.track_id}</code>
                  <span className={`tile__score ${confidenceClass(entry.peakScore)}`}>
                    {entry.peakScore.toFixed(2)}
                  </span>
                </span>
                <span className="tile__sub">
                  {entry.detections.length} views &middot; {entry.matches.length} cand
                </span>
              </button>
            );
          })}
        </div>
        </div>
      )}

      {row ? (
        <section className="result" aria-live="polite">
          <div className="result__head">
            <h3>
              Track <code>t{row.person.track_id}</code> in {result.sources.find((s) => s.id === row.person.video_index)?.label}
            </h3>
            <span className="result__meta">
              {row.person.first_time_s.toFixed(1)}s to {row.person.last_time_s.toFixed(1)}s &middot; peak{" "}
              {row.peakScore.toFixed(3)}
            </span>
          </div>

          <div className="result__cols">
            <div>
              <p className="panel-label">Stored appearances ({row.detections.length})</p>
              <div aria-label="Stored appearances" className="strip" role="group" tabIndex={0}>
                {row.detections.map((detection) => (
                  <figure className="strip__item" key={detection.id}>
                    {detection.crop ? <img alt="" loading="lazy" src={detection.crop} /> : <span className="strip__blank" />}
                    <figcaption>
                      <span>f{detection.frame_index}</span>
                      <span className={confidenceClass(detection.score)}>{detection.score.toFixed(2)}</span>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </div>

            <div>
              <p className="panel-label">
                Ranked candidates in other cameras ({row.matches.length})
              </p>
              {row.matches.length === 0 ? (
                <p className="result__none">
                  Nothing clears a similarity of {settings.similarity.toFixed(2)}. Lower the match
                  threshold to widen the candidate set.
                </p>
              ) : (
                <ol className="candidates">
                  {row.matches.map((match, rank) => (
                    <li className="candidate" key={match.person.id}>
                      <span className="candidate__rank">{rank + 1}</span>
                      {match.person.crop ? (
                        <img alt="" className="candidate__crop" loading="lazy" src={match.person.crop} />
                      ) : (
                        <span className="candidate__crop candidate__crop--none" />
                      )}
                      <span className="candidate__body">
                        <code>v{match.person.video_index + 1}/t{match.person.track_id}</code>
                        <span className="candidate__bar">
                          <span style={{ width: `${Math.max(2, match.similarity * 100)}%` }} />
                        </span>
                      </span>
                      <span className="candidate__score">{match.similarity.toFixed(3)}</span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </div>

          <p className="view__foot view__foot--warn">
            Ranked candidates are reviewable appearance evidence. They are not an identification,
            and the score is not calibrated: in this session non-matches reach 0.998 against matches
            at 0.999.
          </p>
        </section>
      ) : (
        <p className="view__foot">Select anyone in the index to open their appearances and ranked candidates.</p>
      )}
    </div>
  );
}
