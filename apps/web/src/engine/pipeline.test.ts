import { describe, expect, it } from "vitest";

import {
  DEFAULT_SETTINGS,
  RANGES,
  belowIndexFloor,
  DETECTION_STRIDE,
  detectionFrames,
  detectionsAt,
  personFrame,
  runQuery,
  trackBoxesAt,
  trackBoxesForFrame,
  tracksFor,
  type Settings
} from "./pipeline";

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

  it("draws every detection on a frame, not only the indexed people", () => {
    const result = runQuery(base);
    for (const video of result.sources) {
      const frames = detectionFrames(video.id, base);
      expect(frames.length).toBeGreaterThan(0);
      let fromIndex = 0;
      let drawn = 0;
      for (const frame of frames) {
        const boxes = detectionsAt(video.id, frame, base, result);
        drawn += boxes.length;
        for (const row of result.people) {
          if (row.person.video_index !== video.id) continue;
          const views = row.detections.filter((d) => d.frame_index === frame);
          fromIndex += views.length;
          // Every stored index view is one of the boxes drawn on its frame.
          for (const view of views) {
            expect(boxes.some((box) => box.personId === row.person.id)).toBe(true);
          }
        }
        for (const box of boxes) {
          expect(box.score).toBeGreaterThanOrEqual(base.detConf);
          expect(box.bbox[3] - box.bbox[1]).toBeGreaterThanOrEqual(base.minBoxHeight);
        }
      }
      expect(drawn).toBeGreaterThan(fromIndex * 2);
    }
  });

  it("draws the same tracks, with the same ids, as the tracking table counts", () => {
    const result = runQuery(base);
    for (const video of result.sources) {
      const tracks = new Map(tracksFor(video.id, base, result).map((track) => [track.trackId, track]));
      const seen = new Map<number, number>();
      for (let frame = 0; frame < video.frame_count; frame += DETECTION_STRIDE) {
        for (const box of trackBoxesAt(video.id, frame, base)) {
          seen.set(box.trackId, (seen.get(box.trackId) ?? 0) + 1);
        }
      }
      expect(seen.size).toBe(tracks.size);
      for (const [trackId, count] of seen) expect(tracks.get(trackId)?.detections).toBe(count);
    }
  });

  it("places an in-between frame's box halfway between its two observations", () => {
    const video = runQuery(base).sources[0];
    let checked = 0;
    for (let frame = 0; frame + DETECTION_STRIDE < video.frame_count; frame += DETECTION_STRIDE) {
      const before = new Map(trackBoxesAt(video.id, frame, base).map((box) => [box.trackId, box]));
      const after = new Map(trackBoxesAt(video.id, frame + DETECTION_STRIDE, base).map((box) => [box.trackId, box]));
      for (const box of trackBoxesForFrame(video.id, frame + 1, base)) {
        const a = before.get(box.trackId)!;
        const b = after.get(box.trackId);
        // Only tracks seen on the frame before are drawn; nothing is invented.
        expect(a).toBeDefined();
        const expected = b ? a.bbox.map((value, i) => (value + b.bbox[i]) / 2) : a.bbox;
        box.bbox.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 6));
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("seeks a selected person to a frame where their own box is drawn", () => {
    for (const settings of [base, { ...base, detConf: 0.6, minBoxHeight: 60 }]) {
      const result = runQuery(settings);
      expect(result.people.length).toBeGreaterThan(0);
      for (const row of result.people) {
        const frame = personFrame(row);
        const boxes = trackBoxesForFrame(row.person.video_index, frame, settings);
        expect(boxes.some((box) => box.personId === row.person.id), `${row.person.id} at ${frame}`).toBe(true);
      }
    }
  });

  it("stops drawing tracks past the frame limit", () => {
    const limited = { ...base, frameLimit: 30 };
    const video = runQuery(limited).sources[0];
    expect(trackBoxesAt(video.id, 32, limited)).toHaveLength(0);
    expect(trackBoxesForFrame(video.id, 31, limited)).toHaveLength(0);
    expect(trackBoxesAt(video.id, 30, limited).length).toBeGreaterThan(0);
  });
});
