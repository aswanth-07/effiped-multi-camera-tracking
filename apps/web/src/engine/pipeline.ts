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

import { frameDetections, type FrameDetectionRow } from "../data/frameDetections";
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
  // The floor the published tracking numbers were run at (sessionStats.detThresh).
  // Higher hides real people: the nearest pedestrians in camera 2 score 0.40-0.44.
  detConf: 0.35,
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

/**
 * The distribution each threshold is cutting through.
 *
 * A slider that shows only its own number asks the operator to guess what
 * moving it will cost. These are the real stored values behind each control, so
 * the histogram under a slider is the data the threshold acts on, not a
 * decoration of it: the operator can see that the mass of detections sits at
 * 0.60 before deciding to cut at 0.65.
 *
 * Computed once. Nothing here depends on the current settings.
 */
function samplesFor(key: keyof Settings): number[] {
  const appearances = Object.values(personSearch.details).flatMap((entry) => entry.appearances);
  switch (key) {
    case "detConf":
      return appearances.map((appearance) => appearance.score);
    case "trackActivation":
      return personSearch.people.map((person) => person.best_score);
    case "minBoxHeight":
      return appearances.map(boxHeight);
    case "frameLimit":
      return appearances.map((appearance) => appearance.frame_index);
    case "similarity":
      return Object.values(personSearch.matches).flatMap((list) => list.map((m) => m.similarity));
    default:
      return [];
  }
}

export type Distribution = {
  /** Bin heights normalised to 0..1, left to right across the control's range. */
  bins: number[];
  total: number;
};

const BIN_COUNT = 28;

export const DISTRIBUTIONS: Partial<Record<keyof Settings, Distribution>> = Object.fromEntries(
  RANGES.map((range) => {
    const samples = samplesFor(range.key);
    if (samples.length === 0) return [range.key, undefined];
    const bins = new Array(BIN_COUNT).fill(0);
    const span = range.max - range.min || 1;
    for (const value of samples) {
      const slot = Math.round(((value - range.min) / span) * (BIN_COUNT - 1));
      if (slot >= 0 && slot < BIN_COUNT) bins[slot] += 1;
    }
    const peak = Math.max(...bins, 1);
    return [range.key, { bins: bins.map((count) => count / peak), total: samples.length }];
  }).filter((entry): entry is [keyof Settings, Distribution] => entry[1] !== undefined)
) as Partial<Record<keyof Settings, Distribution>>;

/** How many stored values a threshold currently keeps, and out of how many. */
export function survivorsFor(key: keyof Settings, value: number): { kept: number; total: number } | null {
  const samples = samplesFor(key);
  if (samples.length === 0) return null;
  // Frame limit is an upper bound; every other control is a floor.
  const kept =
    key === "frameLimit"
      ? value === 0
        ? samples.length
        : samples.filter((sample) => sample <= value).length
      : samples.filter((sample) => sample >= value).length;
  return { kept, total: samples.length };
}

export function runQuery(settings: Settings): RunResult {
  const selected = new Set(settings.sources);
  const sources = personSearch.videos.filter((video) => selected.has(video.id));
  const frameCap = settings.frameLimit > 0 ? settings.frameLimit : Number.POSITIVE_INFINITY;

  let similaritySum = 0;
  let similarityCount = 0;
  let tracksSeen = 0;

  const people: PersonRow[] = [];

  for (const person of personSearch.people) {
    if (!selected.has(person.video_index)) continue;
    tracksSeen += 1;

    const stored = personSearch.details[person.id]?.appearances ?? [];
    const detections = stored.filter(
      (appearance) =>
        appearance.score >= settings.detConf &&
        boxHeight(appearance) >= settings.minBoxHeight &&
        appearance.frame_index <= frameCap
    );

    // A track with nothing left after filtering is not a person any more, and
    // neither is one that never peaked above the activation threshold.
    if (detections.length === 0) continue;
    const peakScore = Math.max(...detections.map((appearance) => appearance.score));
    if (peakScore < settings.trackActivation) continue;

    const matches = (personSearch.matches[person.id] ?? [])
      .filter((match) => match.similarity >= settings.similarity && selected.has(match.person.video_index))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, settings.topK);

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

  // Detections are counted over every box the detector returned, the same boxes
  // the Detection view draws, not only the few views the person index stores.
  let detectionsKept = 0;
  let detectionsStored = 0;
  let confidenceSum = 0;
  for (const source of sources) {
    for (const [frame, rows] of Object.entries(storedFrames(source.id))) {
      if (Number(frame) > frameCap) continue;
      detectionsStored += rows.length;
      for (const row of keptRows(rows, settings)) {
        detectionsKept += 1;
        confidenceSum += row[4];
      }
    }
  }
  const detectionsDropped = detectionsStored - detectionsKept;
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

/**
 * The frame a person's crop was taken from.
 *
 * Every candidate a reviewer is shown has to be traceable back to a moment in a
 * clip: which camera, which frame, which box. Without that the crop is just a
 * picture, and the whole point of the workflow is that a person can check it.
 */
export type Provenance = {
  video: WorkbenchVideo;
  appearance: WorkbenchAppearance;
};

export function provenanceFor(personId: string): Provenance | null {
  const person = personSearch.people.find((entry) => entry.id === personId);
  if (!person) return null;
  const appearances = personSearch.details[personId]?.appearances ?? [];
  if (appearances.length === 0) return null;
  // Prefer the appearance the headline crop came from, else the most confident.
  const appearance =
    appearances.find((entry) => entry.crop === person.crop) ??
    appearances.reduce((best, entry) => (entry.score > best.score ? entry : best), appearances[0]);
  const video = personSearch.videos.find((entry) => entry.id === person.video_index);
  if (!video) return null;
  return { video, appearance };
}

/** The seconds a clip spans, used to bound the gallery's time filter. */
export function clipDuration(videoId: number): number {
  return personSearch.videos.find((video) => video.id === videoId)?.duration_s ?? 15;
}

export type FrameDetection = {
  bbox: [number, number, number, number];
  score: number;
  trackId: number | null;
  /** The person-search id this box belongs to, when that track survived the run. */
  personId: string | null;
};

/**
 * Which index person each track belongs to. The index merges fragmented tracks
 * into one person, so a person can own several track ids; their stored views
 * carry the original ids.
 */
const PERSON_BY_TRACK = new Map<string, string>();
for (const person of personSearch.people) {
  PERSON_BY_TRACK.set(`${person.video_index}:${person.track_id}`, person.id);
  for (const appearance of personSearch.details[person.id]?.appearances ?? []) {
    PERSON_BY_TRACK.set(`${appearance.video_index}:${appearance.track_id}`, person.id);
  }
}

function storedFrames(videoId: number): Record<string, FrameDetectionRow[]> {
  return frameDetections.videos.find((video) => video.video_index === videoId)?.frames ?? {};
}

function keepsDetection(row: FrameDetectionRow, settings: Settings): boolean {
  return row[4] >= settings.detConf && row[3] - row[1] >= settings.minBoxHeight;
}

/**
 * Processed frames of a clip that still hold a detection at these thresholds.
 *
 * Unlike the person index these are every box the detector returned, so a frame
 * shows everyone in it, not only the people who made it into the index.
 */
export function detectionFrames(videoId: number, settings: Settings): number[] {
  const frameCap = settings.frameLimit > 0 ? settings.frameLimit : Number.POSITIVE_INFINITY;
  return Object.entries(storedFrames(videoId))
    .filter(([frame, rows]) => Number(frame) <= frameCap && rows.some((row) => keepsDetection(row, settings)))
    .map(([frame]) => Number(frame))
    .sort((a, b) => a - b);
}

function overlap(a: FrameDetectionRow, b: FrameDetectionRow): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** Hard NMS at draw time. The decoder uses soft-NMS, which leaves a few decayed
 * duplicates above a low floor; the tracker absorbs them, a drawn frame should not. */
const DRAW_NMS_IOU = 0.5;

function keptRows(rows: FrameDetectionRow[], settings: Settings): FrameDetectionRow[] {
  // Rows are stored strongest first, so greedy suppression keeps the best box.
  const kept: FrameDetectionRow[] = [];
  for (const row of rows) {
    if (!keepsDetection(row, settings)) continue;
    if (kept.some((other) => overlap(row, other) > DRAW_NMS_IOU)) continue;
    kept.push(row);
  }
  return kept;
}

export function detectionsAt(
  videoId: number,
  frame: number,
  settings: Settings,
  result: RunResult | null
): FrameDetection[] {
  return keptRows(storedFrames(videoId)[String(frame)] ?? [], settings).map((row) => {
    const trackId = row[5];
    const id = trackId === null ? null : PERSON_BY_TRACK.get(`${videoId}:${trackId}`) ?? null;
    return {
      bbox: [row[0], row[1], row[2], row[3]],
      score: row[4],
      trackId,
      personId: id !== null && result?.byId[id] ? id : null
    };
  });
}

export type TrackSummary = {
  trackId: number;
  firstFrame: number;
  lastFrame: number;
  /** Detections of this track that survive the current floors. */
  detections: number;
  peakScore: number;
  /** The index person this track feeds, whether or not it survives the thresholds. */
  personId: string | null;
  /** That person's row, when it survives the current thresholds. */
  person: PersonRow | null;
};

/**
 * Every local track the tracker produced in a clip, not only the ones promoted
 * into the person index, so the table agrees with the boxes drawn beside it.
 */
export function tracksFor(videoId: number, settings: Settings, result: RunResult | null): TrackSummary[] {
  const frameCap = settings.frameLimit > 0 ? settings.frameLimit : Number.POSITIVE_INFINITY;
  const tracks = new Map<number, TrackSummary>();
  for (const [key, rows] of Object.entries(storedFrames(videoId))) {
    const frame = Number(key);
    if (frame > frameCap) continue;
    for (const row of keptRows(rows, settings)) {
      const trackId = row[5];
      if (trackId === null) continue;
      const track = tracks.get(trackId);
      if (track) {
        track.firstFrame = Math.min(track.firstFrame, frame);
        track.lastFrame = Math.max(track.lastFrame, frame);
        track.detections += 1;
        track.peakScore = Math.max(track.peakScore, row[4]);
      } else {
        const personId = PERSON_BY_TRACK.get(`${videoId}:${trackId}`) ?? null;
        tracks.set(trackId, {
          trackId,
          firstFrame: frame,
          lastFrame: frame,
          detections: 1,
          peakScore: row[4],
          personId,
          person: (personId && result?.byId[personId]) || null
        });
      }
    }
  }
  return [...tracks.values()].sort((a, b) => a.trackId - b.trackId);
}

/** Frames the indexing run processed: every `DETECTION_STRIDE`th frame of a clip. */
export const DETECTION_STRIDE = frameDetections.generated_with.frame_stride;

export type TrackBox = {
  trackId: number;
  bbox: [number, number, number, number];
  score: number;
  /** The index person this track feeds, when it made it into the index. */
  personId: string | null;
};

/**
 * The tracked boxes on one processed frame that survive the current floors, the
 * same set the Tracking table counts. Empty past the frame limit and on frames
 * the run skipped.
 */
export function trackBoxesAt(videoId: number, frame: number, settings: Settings): TrackBox[] {
  const frameCap = settings.frameLimit > 0 ? settings.frameLimit : Number.POSITIVE_INFINITY;
  if (frame > frameCap) return [];
  const boxes: TrackBox[] = [];
  for (const row of keptRows(storedFrames(videoId)[String(frame)] ?? [], settings)) {
    const trackId = row[5];
    if (trackId === null) continue;
    boxes.push({
      trackId,
      bbox: [row[0], row[1], row[2], row[3]],
      score: row[4],
      personId: PERSON_BY_TRACK.get(`${videoId}:${trackId}`) ?? null
    });
  }
  return boxes;
}

/**
 * Track boxes for the frame a video is showing.
 *
 * The run processed every second frame. On a frame in between, a track seen on
 * both neighbouring processed frames is drawn halfway between them, which is
 * what the tracker's motion model would predict; a track seen on only one side
 * is drawn where it was last observed, never invented.
 */
export function trackBoxesForFrame(videoId: number, frame: number, settings: Settings): TrackBox[] {
  if (settings.frameLimit > 0 && frame > settings.frameLimit) return [];
  const previous = frame - (frame % DETECTION_STRIDE);
  const before = trackBoxesAt(videoId, previous, settings);
  if (previous === frame) return before;
  const after = new Map(trackBoxesAt(videoId, previous + DETECTION_STRIDE, settings).map((box) => [box.trackId, box]));
  const t = (frame - previous) / DETECTION_STRIDE;
  return before.map((box) => {
    const next = after.get(box.trackId);
    if (!next) return box;
    const bbox = box.bbox.map((value, index) => value + (next.bbox[index] - value) * t) as TrackBox["bbox"];
    return { ...box, bbox };
  });
}
