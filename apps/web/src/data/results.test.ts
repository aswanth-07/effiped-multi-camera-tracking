import { describe, expect, it } from "vitest";
import { results } from "./results";

describe("canonical result fixture", () => {
  it("keeps contest, poster, and research evidence separated", () => {
    expect(results.verified_contest_system.pdestre.validation.rank1_cross).toBe(62.8);
    expect(results.verified_contest_system.footprint.pipeline_fps_approx).toBe(18);
    expect(results.contest_submission_snapshot.reported_fps).toBe(22);
    expect(results.post_contest_evolution.partjde.matched_part_readout_gain_pp).toBe(6.66);
    expect(results.post_contest_evolution.boxjde.natural_e2e_rank1_gain_pp).toBe(13.01);
  });

  it("labels the hosted case as a precomputed research replay", () => {
    expect(results.demo_case.description.toLowerCase()).toContain("precomputed");
    expect(results.demo_case.cameras).toHaveLength(4);
  });
});
