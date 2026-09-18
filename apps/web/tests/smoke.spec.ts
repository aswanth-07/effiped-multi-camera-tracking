import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const routes = ["/", "/workbench", "/system", "/evidence", "/deploy"];

test("the overview presents the project and routes into the workbench", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1 })).toContainText("Multi-Camera Pedestrian");
  const eventTerms = ["con" + "test", "compe" + "tition", "pr" + "ize", "aw" + "ard", "SI" + "PC"];
  await expect(page.locator("body")).not.toContainText(new RegExp(eventTerms.join("|"), "i"));

  // Every published figure is read from the evidence fixture, so the headline
  // numbers have to reach the page rather than being written into it.
  await expect(page.locator(".metric-row")).toContainText("62.8");
  await expect(page.locator(".metric-row")).toContainText("7.78");

  // The overview is a route in an application, not a single scrolling page.
  await page.getByRole("link", { name: /open the workbench/i }).first().click();
  await expect(page).toHaveURL(/\/workbench$/);
  await expect(page.getByRole("heading", { name: "EffiPed Pedestrian Tracker" })).toBeVisible();
});

test("client-side routing reaches every route and survives a reload", async ({ page }) => {
  await page.goto("/");
  for (const name of ["System", "Evidence", "Run it", "Overview"]) {
    // Narrow viewports keep the rail behind a drawer, so open it when it is there.
    const toggle = page.getByRole("button", { name: "Open navigation" });
    if (await toggle.isVisible()) await toggle.click();
    await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name }).click();
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  }

  // A deep link has to work on its own, because the rewrite serves the shell.
  await page.goto("/evidence");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Evidence");

  await page.goto("/not-a-real-route");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("No page at");
});

test("the workbench person-search flow works end to end", async ({ page }) => {
  await page.goto("/workbench");

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
  await page.goto("/workbench");
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

test("desktop and mobile layouts do not overflow on any route", async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const route of routes) {
      await page.goto(route);
      const dimensions = await page.evaluate(() => ({
        client: document.documentElement.clientWidth,
        scroll: document.documentElement.scrollWidth
      }));
      expect(dimensions.scroll, `${route} at ${viewport.width}px`).toBe(dimensions.client);
    }
  }
});

test("the mobile drawer opens, navigates and closes on Escape", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  const toggle = page.getByRole("button", { name: "Open navigation" });
  await expect(toggle).toBeVisible();
  await toggle.click();
  await expect(page.locator(".rail.is-open")).toBeVisible();

  await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: "Evidence" }).click();
  await expect(page).toHaveURL(/\/evidence$/);
  await expect(page.locator(".rail.is-open")).toHaveCount(0);

  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(page.locator(".rail.is-open")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator(".rail.is-open")).toHaveCount(0);
});

test("has no serious automated accessibility violations on any route", async ({ page }) => {
  for (const route of routes) {
    await page.goto(route);
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = results.violations.filter((violation) =>
      ["serious", "critical"].includes(violation.impact ?? "")
    );
    expect(serious, `${route}: ${serious.map((v) => v.id).join(", ")}`).toEqual([]);
  }
});

test("honors reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const duration = await page
    .locator(".primary-link")
    .first()
    .evaluate((element) => getComputedStyle(element).transitionDuration);
  expect(Number.parseFloat(duration)).toBeLessThanOrEqual(0.001);
});
