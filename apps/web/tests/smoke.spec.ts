import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

/** Read the live counters the status bar prints for the current job. */
async function counters(page: import("@playwright/test").Page) {
  const text = await page.locator(".statusbar__cells").innerText();
  const read = (name: string) => {
    const match = text.match(new RegExp(`${name}\\s+([0-9.]+)`));
    return match ? Number(match[1]) : Number.NaN;
  };
  return {
    clips: read("clips"),
    detections: read("detections"),
    people: read("people"),
    links: read("links"),
    meanConf: read("mean conf")
  };
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  // The console runs the default job on load.
  await expect(page.locator(".statusbar__stage")).toContainText("ready", { timeout: 15000 });
});

test("boots as a console with a job already loaded", async ({ page }) => {
  await expect(page.locator(".topbar__brand")).toContainText("EffiPed");
  await expect(page.getByRole("tab", { name: "Person Search" })).toHaveAttribute("aria-selected", "true");

  const stats = await counters(page);
  expect(stats.clips).toBe(4);
  expect(stats.people).toBeGreaterThan(10);
  expect(stats.detections).toBeGreaterThan(50);

  // The gallery is the job's output, not a fixed list.
  expect(await page.locator(".tile").count()).toBe(stats.people);

  const eventTerms = ["con" + "test", "compe" + "tition", "pr" + "ize", "aw" + "ard", "SI" + "PC"];
  await expect(page.locator("body")).not.toContainText(new RegExp(eventTerms.join("|"), "i"));
});

test("raising a threshold changes the job output", async ({ page }) => {
  const before = await counters(page);

  await page.locator("#ctl-detConf").fill("0.65");
  await page.locator("#ctl-similarity").fill("0.8");
  await expect(page.locator("output[for='ctl-detConf']")).toHaveText("0.65");
  await expect(page.locator(".controls__state")).toContainText(/changed since the last run/i);
  await page.getByRole("button", { name: /run pipeline|re-run/i }).click();

  // The dirty marker also hides while a run is in flight, so it cannot be the
  // signal. Wait for the counters themselves to move.
  await expect
    .poll(async () => (await counters(page)).detections, { timeout: 15000 })
    .toBeLessThan(before.detections);

  const after = await counters(page);
  expect(after.detections).toBeLessThan(before.detections);
  expect(after.links).toBeLessThan(before.links);
  // Whatever survives a higher floor is more confident than the full set was.
  expect(after.meanConf).toBeGreaterThan(before.meanConf);

  // The readout reports what the change cost, not just where it landed.
  await expect(page.locator(".cell__delta.is-down").first()).toBeVisible();
  await expect(page.locator(".controls__state")).toContainText(/matches the current settings/i);
});

test("each threshold shows the distribution it cuts and what survives", async ({ page }) => {
  const histograms = page.locator(".histogram");
  expect(await histograms.count()).toBeGreaterThanOrEqual(5);

  const note = page.locator("#ctl-detConf-note");
  await expect(note).toContainText(/of .* kept/i);
  const before = await note.innerText();

  await page.locator("#ctl-detConf").fill("0.68");
  await expect(note).not.toHaveText(before);

  // Bars below the cut are marked dead, bars above it live.
  const control = page.locator(".control").filter({ has: page.locator("#ctl-detConf") });
  expect(await control.locator(".histogram__bar.is-live").count()).toBeGreaterThan(0);
  expect(await control.locator(".histogram__bar:not(.is-live)").count()).toBeGreaterThan(0);
});

test("the source picker attaches a subset and the job narrows", async ({ page }) => {
  await page.getByRole("button", { name: /select video sources/i }).click();

  const dialog = page.getByRole("dialog", { name: /select video sources/i });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".source-card")).toHaveCount(4);

  await dialog.getByRole("button", { name: /camera 3/i }).click();
  await dialog.getByRole("button", { name: /camera 4/i }).click();
  await expect(dialog.locator(".modal__count")).toContainText("2 of 4");

  await dialog.getByRole("button", { name: /attach 2 clips/i }).click();
  await expect(dialog).toBeHidden();

  // "ready" is already on screen from the previous job, so it cannot be the
  // signal that this one finished. Wait for the clip count itself.
  await expect.poll(async () => (await counters(page)).clips, { timeout: 15000 }).toBe(2);

  const stats = await counters(page);
  expect(stats.clips).toBe(2);
  // Every indexed person now belongs to one of the two attached clips.
  const codes = await page.locator(".tile code").allInnerTexts();
  expect(codes.length).toBeGreaterThan(0);
  for (const code of codes) expect(code.startsWith("v1/") || code.startsWith("v2/")).toBe(true);
});

test("the picker closes on Escape without changing the job", async ({ page }) => {
  const before = await counters(page);
  await page.getByRole("button", { name: /select video sources/i }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  expect(await counters(page)).toEqual(before);
});

test("selecting a person opens their appearances and ranked candidates", async ({ page }) => {
  await page.locator(".tile").first().click();
  await expect(page.locator(".tile.is-on")).toHaveCount(1);

  const result = page.locator(".result");
  await expect(result).toBeVisible();
  await expect(result.locator(".strip__item").first()).toBeVisible();

  const candidates = result.locator(".candidate");
  await expect(candidates.first()).toBeVisible();

  // Candidates arrive ranked.
  const scores = (await result.locator(".candidate__score").allInnerTexts()).map(Number);
  expect(scores.length).toBeGreaterThan(1);
  expect([...scores].sort((a, b) => b - a)).toEqual(scores);

  await expect(result).toContainText(/not an identification/i);
});

test("every workspace renders and number keys switch between them", async ({ page }) => {
  for (const [key, heading] of [
    ["1", /attached sources/i],
    ["2", /single frame detection/i],
    ["3", /camera local tracking/i],
    ["5", /cross camera association/i],
    ["6", /model and runtime/i],
    ["4", /person search index/i]
  ] as const) {
    await page.keyboard.press(key);
    await expect(page.locator(".viewport").getByRole("heading", { level: 2 })).toHaveText(heading);
  }
});

test("detection draws stored boxes on a decoded frame", async ({ page }) => {
  await page.getByRole("tab", { name: "Detection" }).click();
  const scrub = page.locator("#frame-scrub");
  await expect(scrub).toBeEnabled();

  // Mid-clip, where the footage is lit and the frame is not a dark lead-in.
  const max = Number(await scrub.getAttribute("max"));
  await scrub.fill(String(Math.floor(max / 2)));
  await expect(page.locator(".stage__hud")).toContainText(/\d+ detections/);

  await expect
    .poll(
      async () =>
        page.locator(".stage__canvas").evaluate((element) => {
          const canvas = element as HTMLCanvasElement;
          const ctx = canvas.getContext("2d");
          if (!ctx || canvas.width === 0) return 0;
          const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
          let lit = 0;
          for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 12 || data[i + 1] > 12 || data[i + 2] > 12) lit += 1;
          }
          return Math.round((100 * lit) / (data.length / 4));
        }),
      { timeout: 20000 }
    )
    .toBeGreaterThan(40);
});

test("no layout overflows at desktop or phone width", async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const tab of ["Sources", "Detection", "Tracking", "Person Search", "Cross Camera", "Model"]) {
      await page.getByRole("tab", { name: tab }).click();
      const size = await page.evaluate(() => ({
        client: document.documentElement.clientWidth,
        scroll: document.documentElement.scrollWidth
      }));
      expect(size.scroll, `${tab} at ${viewport.width}px`).toBe(size.client);
    }
  }
});

test("has no serious automated accessibility violations", async ({ page }) => {
  for (const tab of ["Sources", "Detection", "Tracking", "Person Search", "Cross Camera", "Model"]) {
    await page.getByRole("tab", { name: tab }).click();
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = results.violations.filter((violation) =>
      ["serious", "critical"].includes(violation.impact ?? "")
    );
    expect(serious, `${tab}: ${serious.map((v) => v.id).join(", ")}`).toEqual([]);
  }

  await page.getByRole("button", { name: /select video sources/i }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  const modalResults = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  const modalSerious = modalResults.violations.filter((violation) =>
    ["serious", "critical"].includes(violation.impact ?? "")
  );
  expect(modalSerious, `modal: ${modalSerious.map((v) => v.id).join(", ")}`).toEqual([]);
});

test("honors reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await expect(page.locator(".statusbar__stage")).toContainText("ready", { timeout: 15000 });
  const duration = await page
    .locator(".run-button")
    .first()
    .evaluate((element) => getComputedStyle(element).transitionDuration);
  expect(Number.parseFloat(duration)).toBeLessThanOrEqual(0.001);
});
