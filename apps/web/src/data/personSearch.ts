import fixtureJson from "./person-search.json";
import type { PersonSearchFixture, WorkbenchAppearance, WorkbenchPerson } from "../types";

export const personSearch = fixtureJson as unknown as PersonSearchFixture;

/** Real per-session numbers from cross_camera_demo/session_3_4cam/stats.json. */
export const sessionStats = {
  session: "12-11-2019_3",
  maxFrames: 150,
  fps: 10,
  detThresh: 0.35,
  mergeThresh: 0.3,
  overallMota: 78.0,
  perCameraMota: [86.2, 80.2, 70.1, 77.0],
  trackCounts: [36, 72, 205, 54],
  crossCameraIds: 45,
  correctMatches: 60,
  wrongMatches: 18,
  precision: 0.77,
  perPair: [
    { pair: "C1 ↔ C2", correct: 10, wrong: 0, shared: 26, precision: 1.0 },
    { pair: "C1 ↔ C3", correct: 9, wrong: 2, shared: 27, precision: 0.82 },
    { pair: "C1 ↔ C4", correct: 5, wrong: 0, shared: 20, precision: 1.0 },
    { pair: "C2 ↔ C3", correct: 19, wrong: 8, shared: 28, precision: 0.7 },
    { pair: "C2 ↔ C4", correct: 10, wrong: 2, shared: 21, precision: 0.83 },
    { pair: "C3 ↔ C4", correct: 7, wrong: 6, shared: 22, precision: 0.54 }
  ]
} as const;

export function appearancesFor(personId: string): WorkbenchAppearance[] {
  return personSearch.details[personId]?.appearances ?? [];
}

export function matchesFor(personId: string) {
  return personSearch.matches[personId] ?? [];
}

/** The appearance a person's headline crop came from, so we can redraw its frame. */
export function representativeAppearance(person: WorkbenchPerson): WorkbenchAppearance | undefined {
  const appearances = appearancesFor(person.id);
  if (appearances.length === 0) return undefined;
  return appearances.find((item) => item.crop === person.crop) ?? appearances[0];
}

export function videoFor(videoIndex: number) {
  return personSearch.videos.find((video) => video.id === videoIndex) ?? personSearch.videos[0];
}
