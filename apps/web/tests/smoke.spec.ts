import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("award case study and four-camera investigation work", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("3rd Prize · Student Innovation Project Contest 2026")).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Multi-Camera Pedestrian");
  await page.getByRole("link", { name: /Explore the investigation/i }).click();
  await page.getByRole("tab", { name: /C2/ }).click();
  await expect(page.getByRole("tab", { name: /C2/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("Cross-camera candidates")).toBeVisible();
});

test("has no serious automated accessibility violations", async ({ page }) => {
  await page.goto("/");
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(results.violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""))).toEqual([]);
});

test("honors reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const duration = await page.locator(".scan-line").evaluate((element) => getComputedStyle(element).animationDuration);
  expect(Number.parseFloat(duration)).toBeLessThanOrEqual(0.001);
});
