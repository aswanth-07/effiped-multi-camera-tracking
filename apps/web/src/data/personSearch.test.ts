import { describe, expect, it } from "vitest";

import {
  appearancesFor,
  matchesFor,
  personSearch,
  representativeAppearance,
  videoFor
} from "./personSearch";

describe("precomputed person-search fixture", () => {
  it("indexes the four session_3 clips", () => {
    expect(personSearch.videos).toHaveLength(4);
    for (const [index, video] of personSearch.videos.entries()) {
      expect(video.file_name).toBe(`cam${index + 1}.mp4`);
      expect(video.source).toBe(`/media/pdestre/session3/cam${index + 1}-source.webm`);
      expect(video.tracked).toBe(`/media/pdestre/session3/cam${index + 1}-tracked.webm`);
      expect(video.width).toBe(960);
      expect(video.height).toBe(540);
    }
  });

  it("indexes people drawn from every clip", () => {
    expect(personSearch.people.length).toBeGreaterThanOrEqual(20);
    const videoIndices = new Set(personSearch.people.map((person) => person.video_index));
    expect([...videoIndices].sort()).toEqual([0, 1, 2, 3]);
  });

  it("gives every person appearances with drawable geometry", () => {
    for (const person of personSearch.people) {
      const appearances = appearancesFor(person.id);
      expect(appearances.length).toBeGreaterThan(0);
      for (const appearance of appearances) {
        expect(appearance.crop).toMatch(/^\/media\/pdestre\/session3\/crops\/.+\.webp$/);
        expect(appearance.bbox).toHaveLength(4);
        const [x1, y1, x2, y2] = appearance.bbox;
        expect(x2).toBeGreaterThan(x1);
        expect(y2).toBeGreaterThan(y1);
        expect(appearance.time_s).toBeGreaterThanOrEqual(0);
      }
      expect(representativeAppearance(person)).toBeDefined();
    }
  });

  it("ranks cross-video candidates for every person", () => {
    for (const person of personSearch.people) {
      const matches = matchesFor(person.id);
      expect(matches.length).toBeGreaterThan(0);
      // Descending similarity.
      const scores = matches.map((match) => match.similarity);
      expect([...scores].sort((a, b) => b - a)).toEqual(scores);
      // The feature being demonstrated is cross-video association.
      expect(matches.some((match) => !match.same_video)).toBe(true);
      for (const match of matches) {
        expect(match.person.id).not.toBe(person.id);
        expect(personSearch.details[match.person.id]).toBeDefined();
      }
    }
  });

  it("records how the index was produced", () => {
    expect(personSearch.generated_with.checkpoint).toBeTruthy();
    expect(personSearch.generated_with.note.toLowerCase()).toContain("boxjde");
    expect(personSearch.job.status).toBe("complete");
  });

  it("resolves videos by index", () => {
    expect(videoFor(2).label).toBe("Camera 3");
  });
});
