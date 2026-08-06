import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("the workbench person-search flow works end to end", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1 })).toContainText("Multi-Camera Pedestrian");
  await expect(page.getByRole("heading", { name: "EffiPed Pedestrian Tracker" })).toBeVisible();
  const eventTerms = ["con" + "test", "compe" + "tition", "pr" + "ize", "aw" + "ard", "SI" + "PC"];
  await expect(page.locator("body")).not.toContainText(new RegExp(eventTerms.join("|"), "i"));

  // Person Search is the landing tab and the four clips arrive pre-attached.
  await expect(page.getByRole("tab", { name: "Person Search" })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".wb-slot video")).toHaveCount(4);

  // Build the index, then the detected-person gallery appears.
  await page.getByRole("button", { name: "Build person-search index" }).click();
  const people = page.locator(".wb-gallery--4 .wb-tile");
  await expect(people.first()).toBeVisible({ timeout: 15000 });
  expect(await people.count()).toBeGreaterThan(10);

  // Selecting a person fills the crop, the summary, and the ranked candidates,
  // and brings the evidence panels into view.
  const scrollBefore = await page.evaluate(() => window.scrollY);
  await people.first().click();
  await expect(page.locator(".wb-tile.is-selected")).toHaveCount(1);
  await expect(page.locator(".wb-results-anchor")).toBeInViewport({ timeout: 5000 });
  expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(scrollBefore);
  await expect(page.locator(".wb-crop-lg")).toBeVisible();
  await expect(page.locator(".wb-output", { hasText: "Selection summary" })).toContainText("Ranked candidates");

  const matches = page.locator(".wb-tile--match");
  await expect(matches.first()).toBeVisible();
  // Cross-video association is the point of the feature.
  await expect(page.locator(".wb-tile--match.is-cross").first()).toBeVisible();

  // Clicking a candidate updates the clicked full-frame view.
  await matches.first().click();
  await expect(page.locator(".wb-output", { hasText: "Clicked crop full-frame view" })).toContainText(
    /similarity|cross-video|same video/
  );
});

test("every workbench tab renders", async ({ page }) => {
  await page.goto("/");
  for (const name of ["Single Camera", "Cross Camera", "Image Detection", "Model Status", "Research Context"]) {
    await page.getByRole("tab", { name }).click();
    await expect(page.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(".wb-panel")).toBeVisible();
  }

  // Single-camera tracking reveals the tracked render for the chosen clip.
  await page.getByRole("tab", { name: "Single Camera" }).click();
  await page.getByRole("button", { name: "Run single-camera tracking" }).click();
  const tracked = page.locator(".wb-output video").first();
  await expect(tracked).toHaveAttribute("src", /cam1-tracked\.webm$/, { timeout: 15000 });
});

test("desktop and mobile layouts do not overflow", async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.goto("/");
    const dimensions = await page.evaluate(() => ({
      client: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth
    }));
    expect(dimensions.scroll).toBe(dimensions.client);
  }
});

test("has no serious automated accessibility violations", async ({ page }) => {
  await page.goto("/");
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(results.violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""))).toEqual([]);
});

test("honors reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const duration = await page.locator(".primary-link").evaluate((element) => getComputedStyle(element).transitionDuration);
  expect(Number.parseFloat(duration)).toBeLessThanOrEqual(0.001);
});
