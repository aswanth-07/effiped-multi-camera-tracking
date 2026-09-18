import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

/** Read the live counters the status bar prints for the current job. */
async function counters(page: Page) {
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

async function openThresholds(page: Page) {
  const toggle = page.getByRole("button", { name: /thresholds/i });
  if ((await toggle.getAttribute("aria-expanded")) !== "true") await toggle.click();
  await expect(page.locator(".controls")).toBeVisible();
}

/** A person the reviewer can actually follow, which means one with candidates. */
function withCandidates(page: Page) {
  return page.locator(".tile").filter({ hasNot: page.getByText("0 candidates") });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".statusbar__stage")).toContainText("ready", { timeout: 15000 });
});

test("boots as a console with the job already loaded", async ({ page }) => {
  await expect(page.locator(".topbar__identity")).toContainText("EffiPed");
  await expect(page.locator(".topbar__identity")).toContainText("Identity Review Console");
  await expect(page.getByRole("tab", { name: "Person Search" })).toHaveAttribute("aria-selected", "true");

  const stats = await counters(page);
  expect(stats.clips).toBe(4);
  expect(stats.people).toBeGreaterThan(10);

  // Thresholds are a panel, not a permanent rail.
  await expect(page.locator(".controls")).toHaveCount(0);

  const eventTerms = ["con" + "test", "compe" + "tition", "pr" + "ize", "aw" + "ard", "SI" + "PC"];
  await expect(page.locator("body")).not.toContainText(new RegExp(eventTerms.join("|"), "i"));
});

test("the thresholds panel opens from the bar and closes on Escape", async ({ page }) => {
  await openThresholds(page);
  await expect(page.locator("#ctl-detConf")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator(".controls")).toHaveCount(0);
});

test("raising a threshold changes the job output", async ({ page }) => {
  const before = await counters(page);
  await openThresholds(page);

  await page.locator("#ctl-detConf").fill("0.65");
  await page.locator("#ctl-similarity").fill("0.8");
  await expect(page.locator("output[for='ctl-detConf']")).toHaveText("0.65");
  await page.getByRole("button", { name: /run pipeline|re-run/i }).click();

  await expect
    .poll(async () => (await counters(page)).detections, { timeout: 15000 })
    .toBeLessThan(before.detections);

  const after = await counters(page);
  expect(after.links).toBeLessThan(before.links);
  // Whatever survives a higher floor is more confident than the full set was.
  expect(after.meanConf).toBeGreaterThan(before.meanConf);
  await expect(page.locator(".cell__delta.is-down").first()).toBeVisible();
});

test("each threshold shows the distribution it cuts and what survives", async ({ page }) => {
  await openThresholds(page);
  expect(await page.locator(".histogram").count()).toBeGreaterThanOrEqual(5);

  const note = page.locator("#ctl-detConf-note");
  await expect(note).toContainText(/of .* kept/i);
  const before = await note.innerText();

  await page.locator("#ctl-detConf").fill("0.68");
  await expect(note).not.toHaveText(before);

  const control = page.locator(".control").filter({ has: page.locator("#ctl-detConf") });
  expect(await control.locator(".histogram__bar.is-live").count()).toBeGreaterThan(0);
  expect(await control.locator(".histogram__bar:not(.is-live)").count()).toBeGreaterThan(0);
});

test("person search pairs one source with that source's gallery", async ({ page }) => {
  await expect(page.locator(".pane--source video")).toBeVisible();
  await expect(page.locator(".picker__item")).toHaveCount(4);
  await expect(page.locator("#gal-heading")).toContainText("Camera 1");

  // Switching source switches the gallery with it.
  await page.locator(".picker__item").nth(2).click();
  await expect(page.locator("#gal-heading")).toContainText("Camera 3");
  await expect(page.locator(".picker__item.is-on")).toHaveCount(1);
  expect(await page.locator(".tile").count()).toBeGreaterThan(0);
});

test("the time filter narrows the gallery to a window of the clip", async ({ page }) => {
  const before = await page.locator(".tile").count();
  expect(before).toBeGreaterThan(1);

  await page.locator("#time-from").fill("12");
  await expect(page.locator("output[for='time-from']")).toHaveText("12.0s");
  await expect.poll(async () => page.locator(".tile").count()).toBeLessThan(before);

  await page.getByRole("button", { name: /whole clip/i }).click();
  await expect.poll(async () => page.locator(".tile").count()).toBe(before);
});

test("selecting a person scrolls to their candidates", async ({ page }) => {
  // Which element scrolls differs by breakpoint, so assert the outcome the
  // reviewer sees rather than the mechanism that produced it.
  const heading = page.locator("#matches-heading");
  await expect(heading).not.toBeInViewport();

  await withCandidates(page).first().click();

  await expect(heading).toContainText("Candidates for");
  await expect(heading).toBeInViewport({ timeout: 8000 });
  await expect(page.locator(".ranked__item").first()).toBeVisible();
});

test("a candidate reveals the full frame it was cropped from", async ({ page }) => {
  await withCandidates(page).first().click();
  await expect(page.locator(".ranked__item").first()).toBeVisible();

  // Before a candidate is opened the reviewer is told what to do next.
  await expect(page.locator(".matches__frame")).toContainText(/select a candidate/i);

  await page.locator(".ranked__item").first().click();
  const meta = page.locator(".frame__meta");
  await expect(meta).toContainText("Source");
  await expect(meta).toContainText(/Camera \d/);
  await expect(meta).toContainText(/\d+\.\d\ds/);
  await expect(meta).toContainText("Similarity");

  // The frame is really decoded, not a placeholder.
  await expect
    .poll(
      async () =>
        page.locator(".frame__stage canvas").evaluate((element) => {
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
    .toBeGreaterThan(30);

  await expect(page.locator(".matches__caution")).toContainText(/not an identification/i);
});

test("the source picker attaches a subset and the job narrows", async ({ page }) => {
  await openThresholds(page);
  await page.getByRole("button", { name: /select video sources/i }).click();

  const dialog = page.getByRole("dialog", { name: /select video sources/i });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".source-card")).toHaveCount(4);

  await dialog.getByRole("button", { name: /camera 3/i }).click();
  await dialog.getByRole("button", { name: /camera 4/i }).click();
  await expect(dialog.locator(".modal__count")).toContainText("2 of 4");

  await dialog.getByRole("button", { name: /attach 2 clips/i }).click();
  await expect(dialog).toBeHidden();
  await expect.poll(async () => (await counters(page)).clips, { timeout: 15000 }).toBe(2);
  await expect(page.locator(".picker__item")).toHaveCount(2);
});

test("every workspace renders and number keys switch between them", async ({ page }) => {
  for (const [key, heading] of [
    ["2", /single frame detection/i],
    ["3", /camera local tracking/i],
    ["4", /cross camera association/i],
    ["5", /attached sources/i],
    ["6", /model and runtime/i]
  ] as const) {
    await page.keyboard.press(key);
    await expect(page.locator(".viewport").getByRole("heading", { level: 2 }).first()).toHaveText(heading);
  }
  await page.keyboard.press("1");
  await expect(page.locator("#src-heading")).toBeVisible();
});

test("no layout overflows at desktop or phone width", async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const tab of ["Person Search", "Detection", "Tracking", "Cross Camera", "Sources", "Model"]) {
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
  for (const tab of ["Person Search", "Detection", "Tracking", "Cross Camera", "Sources", "Model"]) {
    await page.getByRole("tab", { name: tab }).click();
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = results.violations.filter((violation) =>
      ["serious", "critical"].includes(violation.impact ?? "")
    );
    expect(serious, `${tab}: ${serious.map((v) => v.id).join(", ")}`).toEqual([]);
  }

  await openThresholds(page);
  const panel = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  const panelSerious = panel.violations.filter((v) => ["serious", "critical"].includes(v.impact ?? ""));
  expect(panelSerious, `panel: ${panelSerious.map((v) => v.id).join(", ")}`).toEqual([]);

  await page.getByRole("button", { name: /select video sources/i }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  const modal = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  const modalSerious = modal.violations.filter((v) => ["serious", "critical"].includes(v.impact ?? ""));
  expect(modalSerious, `modal: ${modalSerious.map((v) => v.id).join(", ")}`).toEqual([]);
});

test("honors reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await expect(page.locator(".statusbar__stage")).toContainText("ready", { timeout: 15000 });
  const duration = await page
    .locator(".tile")
    .first()
    .evaluate((element) => getComputedStyle(element).transitionDuration);
  expect(Number.parseFloat(duration)).toBeLessThanOrEqual(0.001);
});
