import { describe, expect, it } from "vitest";
import { results } from "./results";

describe("canonical result fixture", () => {
  it("keeps benchmark and research evidence machine-readable", () => {
    expect(results.system_benchmarks.pdestre.validation.rank1_cross).toBe(62.8);
    expect(results.system_benchmarks.footprint.pipeline_fps_approx).toBe(18);
    expect(results.research_extensions.partjde.matched_part_readout_gain_pp).toBe(6.66);
    expect(results.research_extensions.boxjde.natural_e2e_rank1_gain_pp).toBe(13.01);
  });

  it("contains real replay videos and retrieval candidates", () => {
    expect(results.demo_case.description.toLowerCase()).toContain("precomputed");
    expect(results.demo_case.videos).toHaveLength(2);
    expect(results.demo_case.subjects).toHaveLength(3);
    expect(results.demo_case.subjects[0].candidates).toHaveLength(4);
  });
});
