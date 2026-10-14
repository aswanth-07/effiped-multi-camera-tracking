import fixtureJson from "./person-search.json";
import type { PersonSearchFixture, WorkbenchAppearance, WorkbenchPerson } from "../types";

export const personSearch = fixtureJson as unknown as PersonSearchFixture;

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
