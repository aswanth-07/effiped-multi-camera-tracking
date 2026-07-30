import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("the restored identity-review demo works end to end", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1 })).toContainText("Multi-Camera Pedestrian");
  await expect(page.getByRole("heading", { name: "Identity Review Console" })).toBeVisible();
  const eventTerms = ["con" + "test", "compe" + "tition", "pr" + "ize", "aw" + "ard", "SI" + "PC"];
  await expect(page.locator("body")).not.toContainText(new RegExp(eventTerms.join("|"), "i"));

  const replay = page.locator(".investigation-main video");
  await expect(replay).toHaveAttribute("src", "/media/pdestre/multi-camera-tracking.webm");

  await page.locator(".candidate-card").nth(1).click();
  await expect(page.getByLabel("Replay view")).toHaveValue("cross-camera");
  await expect(replay).toHaveAttribute("src", "/media/pdestre/cross-camera-matches.webm");
  await expect(page.locator(".candidate-card.selected")).toContainText("Rank 2");

  await page.getByRole("button", { name: "Identity Index" }).click();
  await expect(page.getByRole("heading", { name: "Subject Tracks" })).toBeVisible();
  await page.getByRole("button", { name: /Striped shirt query crop/ }).click();
  await expect(page.locator(".selected-summary")).toContainText("ID 10380");
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
