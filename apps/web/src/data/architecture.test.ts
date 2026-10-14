import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { CONFIG, DETECTION_LOSSES, DIAGRAM, IDENTITY_LOSSES, RESULTS } from "./architecture";

/**
 * Flatten the scalar entries of the system configuration into dotted keys.
 * The file is plain nested mappings, so indentation alone gives the path.
 */
function configScalars(text: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const path: { indent: number; key: string }[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/\s+#.*$/, "");
    const match = /^(\s*)([A-Za-z0-9_]+):\s*(.*)$/.exec(line);
    if (!match) continue;
    const indent = match[1].length;
    while (path.length && path[path.length - 1].indent >= indent) path.pop();
    const key = [...path.map((part) => part.key), match[2]].join(".");
    const value = match[3].trim();
    if (value === "") {
      path.push({ indent, key: match[2] });
    } else {
      out[key] = value === "true" ? true : value === "false" ? false : Number.isNaN(Number(value)) ? value : Number(value);
    }
  }
  return out;
}

const config = configScalars(
  readFileSync(new URL("../../../../configs/system/effiped.yaml", import.meta.url), "utf-8")
);

describe("architecture tab", () => {
  it("prints every hyperparameter exactly as the system configuration sets it", () => {
    for (const [key, value] of Object.entries(CONFIG)) {
      expect(config[key], key).toBe(value);
    }
  });

  it("weights each loss term with its configured weight", () => {
    for (const term of [...DETECTION_LOSSES, ...IDENTITY_LOSSES]) {
      expect(term.weight, term.name).toBe(config[term.key]);
    }
  });

  it("draws the compact two-scale trunk the configuration builds", () => {
    // Without BiFPN the backbone stops at stride 8; a diagram showing
    // stride-16 or stride-32 stages would describe a different model.
    expect(config["model.use_bifpn"]).toBe(false);
    expect(config["model.reid_extraction"]).toBe("part_based");
    expect(config["loss.uncertainty_loss"]).toBe(true);
    const svg = readFileSync(new URL(`../../../../docs${DIAGRAM.src}`, import.meta.url), "utf-8");
    expect(svg).toContain(`viewBox="0 0 ${DIAGRAM.width} ${DIAGRAM.height}"`);
    expect(svg).toContain("stem + first two stages");
    expect(config["model.roi_output_size"]).toBe("[32, 8]");
    expect(svg).toContain("RoIAlign 32 × 8");
    expect(svg).not.toMatch(/\/(16|32)\b/);
  });

  it("reads the reported results from the evidence file", () => {
    const text = RESULTS.map(([, value]) => value).join(" ");
    for (const figure of ["62.8", "61.3", "90.74", "88.4", "64.08", "74.24", "61.34", "7.78"]) {
      expect(text).toContain(figure);
    }
  });
});
