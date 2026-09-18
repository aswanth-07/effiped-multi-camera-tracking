/**
 * The replay engine.
 *
 * Inference is the only stage that was removed. Every detection, descriptor,
 * track and pairwise similarity the real pipeline produced is checked in, so a
 * threshold here does what a threshold does in the running system: it decides
 * what survives. Move a slider and the counts, the gallery and the ranked
 * candidates all change, because they are recomputed from the stored records
 * rather than read from a stored answer.
 *
 * What this cannot do is produce a detection the model never made. Lowering the
 * confidence floor below the value the index was built at surfaces nothing new,
 * and the interface says so rather than pretending otherwise.
 */

import { personSearch } from "../data/personSearch";
import type { WorkbenchAppearance, WorkbenchMatch, WorkbenchPerson, WorkbenchVideo } from "../types";

export type Settings = {
  /** Video ids kept in the job. */
  sources: number[];
  /** Detection confidence floor, applied to every stored detection. */
  detConf: number;
  /** A track has to peak above this to be promoted to a person. */
  trackActivation: number;
  /** Minimum box height in source pixels. */
  minBoxHeight: number;
  /** Frames beyond this index are dropped. 0 means the whole clip. */
  frameLimit: number;
  /** Cosine similarity floor for a cross-camera candidate. */
  similarity: number;
  /** Candidates kept per person after ranking. */
  topK: number;
};

export type Range = {
  key: keyof Settings;
  label: string;
  min: number;
  max: number;
  step: number;
  /** Where the checked-in index was actually built. */
  indexedAt?: number;
  unit?: string;
  hint: string;
};

/** The floors the checked-in index was produced at, from `generated_with`. */
export const INDEX_SETTINGS = personSearch.generated_with.settings;

export const DEFAULT_SETTINGS: Settings = {
  sources: [0, 1, 2, 3],
  detConf: 0.45,
  trackActivation: 0.62,
  minBoxHeight: 48,
  frameLimit: 0,
  similarity: 0.35,
  topK: 8
};

export const RANGES: Range[] = [
  {
    key: "detConf",
    label: "Detection confidence",
    min: 0.3,
    max: 0.72,
    step: 0.01,
    indexedAt: 0.31,
    hint: "Centre-point score floor. The index holds detections from 0.31 up, so nothing exists below that."
  },
  {
    key: "trackActivation",
    label: "Track activation",
    min: 0.6,
    max: 0.72,
    step: 0.005,
    indexedAt: 0.61,
    hint: "A track is promoted to a person once its best detection clears this."
  },
  {
    key: "minBoxHeight",
    label: "Minimum box height",
    min: 40,
    max: 180,
    step: 2,
    unit: "px",
    hint: "Drops boxes too short to carry usable appearance. Source frames are 960 by 540."
  },
  {
    key: "frameLimit",
    label: "Frame limit",
    min: 0,
    max: 150,
    step: 5,
    hint: "Stop after this many frames per clip. 0 processes all 150."
  },
  {
    key: "similarity",
    label: "Match similarity",
    min: 0,
    max: 0.92,
    step: 0.01,
    hint: "Cosine floor for a cross-camera candidate. Stored pairs run from -0.05 to 0.92."
  },
  {
    key: "topK",
    label: "Candidates per person",
    min: 1,
    max: 12,
    step: 1,
    hint: "How many ranked candidates are kept after the similarity floor."
  }
];

export type PersonRow = {
  person: WorkbenchPerson;
  /** Detections that survived the current thresholds. */
  detections: WorkbenchAppearance[];
  /** Candidates that survived, already ranked. */
  matches: WorkbenchMatch[];
  peakScore: number;
};

export type RunResult = {
  sources: WorkbenchVideo[];
  people: PersonRow[];
  byId: Record<string, PersonRow>;
  stats: {
    clips: number;
    framesProcessed: number;
    detectionsKept: number;
    detectionsDropped: number;
    tracksSeen: number;
    peopleIndexed: number;
    crossLinks: number;
    meanConfidence: number;
    meanSimilarity: number;
    descriptorDim: number;
  };
};

function boxHeight(appearance: WorkbenchAppearance): number {
  return appearance.bbox[3] - appearance.bbox[1];
}

export function runQuery(settings: Settings): RunResult {
  const selected = new Set(settings.sources);
  const sources = personSearch.videos.filter((video) => selected.has(video.id));
  const frameCap = settings.frameLimit > 0 ? settings.frameLimit : Number.POSITIVE_INFINITY;

  let detectionsDropped = 0;
  let confidenceSum = 0;
  let similaritySum = 0;
  let similarityCount = 0;
  let tracksSeen = 0;

  const people: PersonRow[] = [];

  for (const person of personSearch.people) {
    if (!selected.has(person.video_index)) continue;
    tracksSeen += 1;

    const stored = personSearch.details[person.id]?.appearances ?? [];
    const detections = stored.filter((appearance) => {
      const keep =
        appearance.score >= settings.detConf &&
        boxHeight(appearance) >= settings.minBoxHeight &&
        appearance.frame_index <= frameCap;
      if (!keep) detectionsDropped += 1;
      return keep;
    });

    // A track with nothing left after filtering is not a person any more, and
    // neither is one that never peaked above the activation threshold.
    if (detections.length === 0) continue;
    const peakScore = Math.max(...detections.map((appearance) => appearance.score));
    if (peakScore < settings.trackActivation) continue;

    const matches = (personSearch.matches[person.id] ?? [])
      .filter((match) => match.similarity >= settings.similarity && selected.has(match.person.video_index))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, settings.topK);

    for (const appearance of detections) confidenceSum += appearance.score;
    for (const match of matches) {
      similaritySum += match.similarity;
      similarityCount += 1;
    }

    people.push({ person, detections, matches, peakScore });
  }

  // Strongest first: a reviewer should meet the most confident tracks at the top.
  people.sort((a, b) => b.peakScore - a.peakScore);

  const byId: Record<string, PersonRow> = {};
  for (const row of people) byId[row.person.id] = row;

  const detectionsKept = people.reduce((total, row) => total + row.detections.length, 0);
  const crossLinks = people.reduce(
    (total, row) => total + row.matches.filter((match) => !match.same_video).length,
    0
  );
  const perClipFrames = Math.min(frameCap === Number.POSITIVE_INFINITY ? 150 : frameCap, 150);

  return {
    sources,
    people,
    byId,
    stats: {
      clips: sources.length,
      framesProcessed: perClipFrames * sources.length,
      detectionsKept,
      detectionsDropped,
      tracksSeen,
      peopleIndexed: people.length,
      crossLinks,
      meanConfidence: detectionsKept > 0 ? confidenceSum / detectionsKept : 0,
      meanSimilarity: similarityCount > 0 ? similaritySum / similarityCount : 0,
      descriptorDim: 256
    }
  };
}

/** True when a setting asks for something the stored index cannot contain. */
export function belowIndexFloor(settings: Settings): Range[] {
  return RANGES.filter((range) => {
    if (range.indexedAt === undefined) return false;
    return (settings[range.key] as number) < range.indexedAt;
  });
}
