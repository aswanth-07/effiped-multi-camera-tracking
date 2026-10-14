// Rasterise docs/architecture/effiped-architecture.svg to a 2x PNG with the
// Chromium that the web app's browser tests already install.
//
//   node tools/architecture/export_png.mjs [scale]
//
// Run `npm ci --prefix apps/web` and `npx --prefix apps/web playwright install
// chromium` first if those are not present.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const require = createRequire(resolve(root, "apps", "web", "package.json"));
const { chromium } = require("@playwright/test");

const svgPath = resolve(root, "docs", "architecture", "effiped-architecture.svg");
const pngPath = resolve(root, "docs", "architecture", "effiped-architecture.png");
const scale = Number(process.argv[2] ?? 2);

const svg = readFileSync(svgPath, "utf-8");
const [, width, height] = svg.match(/viewBox="0 0 (\d+) (\d+)"/).map(Number);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: scale });
await page.goto(pathToFileURL(svgPath).href);
await page.screenshot({ path: pngPath, clip: { x: 0, y: 0, width, height } });
await browser.close();
console.log(`wrote ${pngPath} (${width * scale} x ${height * scale})`);
