import { Images, MousePointerClick, Search, UserRound, Users } from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";

import {
  appearancesFor,
  matchesFor,
  personSearch,
  representativeAppearance,
  videoFor
} from "../data/personSearch";
import type { WorkbenchAppearance, WorkbenchMatch, WorkbenchPerson } from "../types";
import { FrameView } from "./FrameView";
import {
  ModelPicker,
  OutputBox,
  Placeholder,
  RunButton,
  SettingsAccordion,
  VideoSlot,
  useSimulatedRun,
  useSliders,
  type SliderSpec
} from "./workbenchControls";

const indexingSliders: SliderSpec[] = [
  { label: "Decode threshold", min: 0.01, max: 0.5, step: 0.01, value: 0.05 },
  { label: "Track activation threshold", min: 0.1, max: 0.8, step: 0.05, value: 0.3 },
  { label: "Frame limit", min: 20, max: 1200, step: 20, value: 240 },
  { label: "Process every Nth frame", min: 1, max: 20, step: 1, value: 2 },
  { label: "Maximum people shown", min: 10, max: 300, step: 10, value: 40 },
  { label: "Views stored per person", min: 2, max: 24, step: 1, value: 6 },
  { label: "Top-K detections", min: 50, max: 600, step: 50, value: 300 }
];

const runSteps = [
  { at: 0, message: "Decoding source clips" },
  { at: 0.25, message: "Running detection" },
  { at: 0.5, message: "Associating tracks" },
  { at: 0.72, message: "Extracting identity descriptors" },
  { at: 0.9, message: "Building gallery index" }
];

type Clicked = {
  title: string;
  caption: string;
  videoIndex: number;
  timeS: number;
  bbox: [number, number, number, number];
};

function clickedFromAppearance(appearance: WorkbenchAppearance, person: WorkbenchPerson): Clicked {
  return {
    title: `ID ${person.id} · frame ${appearance.frame_index}`,
    caption: appearance.caption,
    videoIndex: appearance.video_index,
    timeS: appearance.time_s,
    bbox: appearance.bbox
  };
}

function clickedFromMatch(match: WorkbenchMatch): Clicked | null {
  const appearance = representativeAppearance(match.person);
  if (!appearance) return null;
  return {
    title: `Match ID ${match.person.id} · similarity ${match.similarity.toFixed(3)}`,
    caption: `${match.person.caption} · ${match.same_video ? "same video" : "cross-video match"}`,
    videoIndex: match.person.video_index,
    timeS: appearance.time_s,
    bbox: appearance.bbox
  };
}

export function PersonSearchTab({
  modelKey,
  onModelChange,
  modelOptions
}: {
  modelKey: string;
  onModelChange: (next: string) => void;
  modelOptions: { key: string; label: string; available: boolean }[];
}) {
  const sliders = useSliders(indexingSliders);
  const run = useSimulatedRun(runSteps, 2600);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [clicked, setClicked] = useState<Clicked | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);

  const indexed = run.state === "done";
  const people = personSearch.people;

  const selected = useMemo(
    () => people.find((person) => person.id === selectedId) ?? null,
    [people, selectedId]
  );
  const appearances = selected ? appearancesFor(selected.id) : [];
  const matches = selected ? matchesFor(selected.id) : [];
  const headline = selected ? representativeAppearance(selected) : undefined;

  const selectPerson = useCallback((person: WorkbenchPerson) => {
    setSelectedId(person.id);
    const appearance = representativeAppearance(person);
    setClicked(appearance ? clickedFromAppearance(appearance, person) : null);

    // The evidence panels sit below the gallery, so a click would otherwise
    // update content the visitor cannot see. Wait a frame for the panels to
    // mount on the first selection before scrolling to them.
    window.requestAnimationFrame(() => {
      const target = resultsRef.current;
      if (!target) return;
      const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    });
  }, []);

  const crossVideoCount = matches.filter((match) => !match.same_video).length;

  return (
    <div className="wb-tab wb-tab--search">
      <div className="wb-row wb-row--split">
        <div className="wb-col wb-col--controls">
          <p className="wb-lede">
            Process up to four videos, inspect every detected track, then click a person to view their
            appearances and nearest cross-video matches.
          </p>

          <div className="wb-slot-grid">
            {personSearch.videos.map((video, index) => (
              <VideoSlot
                fileName={video.file_name}
                key={video.id}
                label={`Video ${index + 1}`}
                optional={index > 1}
                poster={video.poster}
                src={video.source}
              />
            ))}
          </div>

          <ModelPicker onChange={onModelChange} options={modelOptions} value={modelKey} />

          <SettingsAccordion
            onChange={sliders.set}
            specs={indexingSliders}
            title="Indexing settings"
            values={sliders.values}
          />

          <RunButton label="Build person-search index" run={run} runningLabel="Indexing…" />

          <OutputBox label="Index status" lines={5}>
            {indexed ? (
              <pre>
{`Indexed ${personSearch.job.people_count} people across ${personSearch.videos.length} videos.
Model: ${personSearch.generated_with.checkpoint}
Frame sets processed: ${personSearch.job.processed_frame_sets}/${personSearch.job.total_frame_sets}
Status: ${personSearch.job.status} — ${personSearch.job.message}`}
              </pre>
            ) : (
              <Placeholder text="Build the index to populate the detected-person gallery." />
            )}
          </OutputBox>
        </div>

        <div className="wb-col wb-col--wide">
          <OutputBox label={`Detected people${indexed ? ` (${people.length})` : ""}`}>
            {indexed ? (
              <>
                <div className="wb-gallery wb-gallery--4">
                  {people.map((person) => (
                    <button
                      aria-pressed={person.id === selectedId}
                      className={`wb-tile ${person.id === selectedId ? "is-selected" : ""}`}
                      key={person.id}
                      onClick={() => selectPerson(person)}
                      type="button"
                    >
                      {person.crop ? <img alt={`Detected person ${person.id}`} loading="lazy" src={person.crop} /> : null}
                      <span className="wb-tile__cap">
                        <strong>{person.id}</strong>
                        <small>V{person.video_index + 1} · {person.num_samples} views</small>
                      </span>
                    </button>
                  ))}
                </div>
                <p className="wb-hint">
                  <MousePointerClick size={13} /> Click any detected person to inspect appearances and
                  similar tracks.
                </p>
              </>
            ) : (
              <Placeholder text="No index yet. Run “Build person-search index” to detect people across the four clips." />
            )}
          </OutputBox>
        </div>
      </div>

      {indexed ? (
        <>
          <div className="wb-row wb-row--split wb-results-anchor" ref={resultsRef}>
            <div className="wb-col">
              <OutputBox label="Selected crop">
                {selected?.crop ? (
                  <img alt={`Selected person ${selected.id}`} className="wb-crop-lg" src={selected.crop} />
                ) : (
                  <Placeholder text="Select a person above." />
                )}
              </OutputBox>
              <OutputBox label="Selection summary" lines={5}>
                {selected ? (
                  <pre>
{`Person: ${selected.id}
Source: ${videoFor(selected.video_index).label} (${videoFor(selected.video_index).file_name})
Track: ${selected.track_id}
Views stored: ${appearances.length} of ${selected.num_samples} detections
Visible: ${selected.first_time_s.toFixed(1)}s – ${selected.last_time_s.toFixed(1)}s
Best detection score: ${selected.best_score.toFixed(3)}
Ranked candidates: ${matches.length} (${crossVideoCount} cross-video)`}
                  </pre>
                ) : (
                  <Placeholder text="Select a person above." />
                )}
              </OutputBox>
            </div>

            <div className="wb-col wb-col--wide">
              <OutputBox label="Selected full-frame view">
                {selected && headline ? (
                  <FrameView
                    alt={`Full frame showing person ${selected.id}`}
                    box={headline.bbox}
                    boxLabel={`ID ${selected.id}`}
                    src={videoFor(selected.video_index).source}
                    timeS={headline.time_s}
                    width={880}
                  />
                ) : (
                  <Placeholder text="Select a person to see where they appear in the source frame." />
                )}
              </OutputBox>

              <OutputBox label="Clicked crop full-frame view">
                {clicked ? (
                  <>
                    <FrameView
                      alt={clicked.title}
                      box={clicked.bbox}
                      boxLabel={clicked.title}
                      dim
                      src={videoFor(clicked.videoIndex).source}
                      timeS={clicked.timeS}
                      width={880}
                    />
                    <p className="wb-caption">{clicked.caption}</p>
                  </>
                ) : (
                  <Placeholder text="Click an appearance or a ranked candidate below." />
                )}
              </OutputBox>
            </div>
          </div>

          <div className="wb-row wb-row--split">
            <div className="wb-col">
              <OutputBox label="All stored appearances for selected person">
                {selected ? (
                  <div className="wb-gallery wb-gallery--5">
                    {appearances.map((appearance) => (
                      <button
                        className="wb-tile"
                        key={appearance.id}
                        onClick={() => setClicked(clickedFromAppearance(appearance, selected))}
                        type="button"
                      >
                        {appearance.crop ? (
                          <img alt={appearance.caption} loading="lazy" src={appearance.crop} />
                        ) : null}
                        <span className="wb-tile__cap">
                          <strong>{appearance.time_s.toFixed(1)}s</strong>
                          <small>{appearance.score.toFixed(2)}</small>
                        </span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <Placeholder text="Select a person above." />
                )}
              </OutputBox>
            </div>

            <div className="wb-col">
              <OutputBox label="Full-frame appearances">
                {selected ? (
                  <div className="wb-gallery wb-gallery--2">
                    {appearances.map((appearance) => (
                      <button
                        className="wb-tile wb-tile--frame"
                        key={appearance.id}
                        onClick={() => setClicked(clickedFromAppearance(appearance, selected))}
                        type="button"
                      >
                        <FrameView
                          alt={`Frame ${appearance.frame_index} for ${selected.id}`}
                          box={appearance.bbox}
                          src={videoFor(appearance.video_index).source}
                          timeS={appearance.time_s}
                          width={360}
                        />
                        <span className="wb-tile__cap">
                          <strong>frame {appearance.frame_index}</strong>
                          <small>{appearance.time_s.toFixed(1)}s</small>
                        </span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <Placeholder text="Select a person above." />
                )}
              </OutputBox>
            </div>
          </div>

          <OutputBox
            label={`Similar detected tracks across uploaded videos${
              selected ? ` — ${matches.length} candidates` : ""
            }`}
          >
            {selected ? (
              matches.length > 0 ? (
                <>
                  <div className="wb-gallery wb-gallery--5">
                    {matches.map((match) => {
                      const target = clickedFromMatch(match);
                      return (
                        <button
                          className={`wb-tile wb-tile--match ${match.same_video ? "is-same" : "is-cross"}`}
                          key={match.person.id}
                          onClick={() => target && setClicked(target)}
                          type="button"
                        >
                          <span className="wb-score">{match.similarity.toFixed(3)}</span>
                          {match.person.crop ? (
                            <img alt={`Candidate ${match.person.id}`} loading="lazy" src={match.person.crop} />
                          ) : null}
                          <span className="wb-tile__cap">
                            <strong>{match.person.id}</strong>
                            <small>
                              {match.same_video
                                ? `same video · V${match.person.video_index + 1}`
                                : `cross-video · V${match.person.video_index + 1}`}
                            </small>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  <p className="wb-hint">
                    <Users size={13} /> Ranked by identity-descriptor cosine similarity. Candidate scores
                    rank visual evidence and are not proof of identity.
                  </p>
                </>
              ) : (
                <Placeholder text="No candidates above threshold for this person." />
              )
            ) : (
              <Placeholder text="Select a person to rank their nearest tracks across the other videos." />
            )}
          </OutputBox>

          <p className="wb-provenance">
            <Search size={13} /> {personSearch.generated_with.note}
          </p>
        </>
      ) : (
        <div className="wb-empty">
          <UserRound size={22} />
          <strong>The four clips are already attached.</strong>
          <span>
            Run the index to detect people, then select anyone to review their appearances and ranked
            cross-video candidates.
          </span>
          <Images aria-hidden="true" size={16} />
        </div>
      )}
    </div>
  );
}
