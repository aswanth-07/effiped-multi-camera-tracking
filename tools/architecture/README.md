# Architecture diagram

`docs/architecture/effiped-architecture.svg` and `.png` are generated, not
hand-edited. `build_diagram.py` is the source: it lays out every block, arrow
and label explicitly and writes the SVG on the console's black ground, so the
Architecture tab can show it unchanged.

## Rebuild

```bash
python tools/architecture/build_diagram.py   # writes the SVG
node tools/architecture/export_png.mjs        # 2x PNG, using the web app's Playwright Chromium
python tools/validate_release.py              # the SVG must keep its <title> and <desc>
```

`export_png.mjs` needs `npm ci --prefix apps/web` and, once,
`npx --prefix apps/web playwright install chromium`.

## Conventions

- Data flows from the bottom up. A grey group is one module; its caption names it.
- Block colour is the kind of operation: blue convolution, purple projection or
  sampling, amber attention or deformable convolution, yellow normalisation, green
  output, grey a fixed operation, rose the input stem. The legend in the corner
  repeats this.
- Dashed `L_det` and `L_id` tags mark where the training losses attach; the
  Architecture tab lists every term and weight.

## Keeping it true

Every layer and number is taken from `configs/system/effiped.yaml` and
`src/effiped`. `apps/web/src/data/architecture.test.ts` checks the configuration
values the tab prints and fails if the diagram ever shows a stride-16 or
stride-32 stage, which this compact two-scale trunk does not have. Change the
model, then change `build_diagram.py` and rebuild.
