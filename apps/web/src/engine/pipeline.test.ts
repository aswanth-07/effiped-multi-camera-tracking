import { describe, expect, it } from "vitest";

import { DEFAULT_SETTINGS, RANGES, belowIndexFloor, runQuery, type Settings } from "./pipeline";

const base: Settings = { ...DEFAULT_SETTINGS };

describe("replay engine", () => {
  it("indexes people from every attached clip by default", () => {
    const result = runQuery(base);
    expect(result.sources).toHaveLength(4);
    expect(result.people.length).toBeGreaterThan(10);
    expect(result.stats.descriptorDim).toBe(256);
    // Strongest first, so a reviewer meets the most confident tracks at the top.
    const peaks = result.people.map((row) => row.peakScore);
    expect([...peaks].sort((a, b) => b - a)).toEqual(peaks);
  });

  it("drops detections as the confidence floor rises", () => {
    const low = runQuery({ ...base, detConf: 0.35 });
    const high = runQuery({ ...base, detConf: 0.65 });
    expect(high.stats.detectionsKept).toBeLessThan(low.stats.detectionsKept);
    // The survivors of a higher floor are, by construction, more confident.
    expect(high.stats.meanConfidence).toBeGreaterThan(low.stats.meanConfidence);
  });

  it("drops cross-camera links as the similarity floor rises", () => {
    const loose = runQuery({ ...base, similarity: 0.1 });
    const strict = runQuery({ ...base, similarity: 0.85 });
    expect(strict.stats.crossLinks).toBeLessThan(loose.stats.crossLinks);
    for (const row of strict.people) {
      for (const match of row.matches) expect(match.similarity).toBeGreaterThanOrEqual(0.85);
    }
  });

  it("keeps at most topK candidates per person", () => {
    const result = runQuery({ ...base, similarity: 0, topK: 3 });
    for (const row of result.people) expect(row.matches.length).toBeLessThanOrEqual(3);
  });

  it("narrows the job to the attached clips only", () => {
    const two = runQuery({ ...base, sources: [0, 1] });
    expect(two.sources.map((video) => video.id)).toEqual([0, 1]);
    for (const row of two.people) expect([0, 1]).toContain(row.person.video_index);
    // A candidate in a detached clip is not a candidate.
    for (const row of two.people) {
      for (const match of row.matches) expect([0, 1]).toContain(match.person.video_index);
    }
  });

  it("returns an empty job when nothing is attached", () => {
    const none = runQuery({ ...base, sources: [] });
    expect(none.people).toHaveLength(0);
    expect(none.stats.detectionsKept).toBe(0);
    expect(none.stats.meanConfidence).toBe(0);
  });

  it("applies the box height floor to stored boxes", () => {
    const tall = runQuery({ ...base, minBoxHeight: 150 });
    for (const row of tall.people) {
      for (const detection of row.detections) {
        expect(detection.bbox[3] - detection.bbox[1]).toBeGreaterThanOrEqual(150);
      }
    }
  });

  it("truncates by frame limit", () => {
    const early = runQuery({ ...base, frameLimit: 30 });
    for (const row of early.people) {
      for (const detection of row.detections) expect(detection.frame_index).toBeLessThanOrEqual(30);
    }
    expect(early.stats.framesProcessed).toBe(30 * 4);
  });

  it("names a threshold that asks for less than the index holds", () => {
    expect(belowIndexFloor(base)).toHaveLength(0);
    const under = belowIndexFloor({ ...base, detConf: 0.3 });
    expect(under.map((range) => range.key)).toContain("detConf");
  });

  it("declares a range for every threshold the interface exposes", () => {
    for (const range of RANGES) {
      expect(range.min).toBeLessThan(range.max);
      expect(typeof DEFAULT_SETTINGS[range.key]).toBe("number");
      const value = DEFAULT_SETTINGS[range.key] as number;
      expect(value).toBeGreaterThanOrEqual(range.min);
      expect(value).toBeLessThanOrEqual(range.max);
    }
  });
});
