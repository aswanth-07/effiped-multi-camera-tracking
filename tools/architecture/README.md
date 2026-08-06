# Architecture diagram

`docs/architecture/effiped-architecture.{pptx,svg,png}` are generated, not hand-edited.
The PowerPoint is the source of truth; the SVG and PNG are exported from it.

## Rebuild

```bash
npm install pptxgenjs                     # ad-hoc; not a repo dependency
node tools/architecture/build_architecture.mjs docs/architecture/effiped-architecture.pptx

# PNG (2560 x 1440)
soffice --headless --convert-to pdf --outdir /tmp docs/architecture/effiped-architecture.pptx
pdftoppm -png -r 192 /tmp/effiped-architecture.pdf /tmp/arch
cp /tmp/arch-1.png docs/architecture/effiped-architecture.png

# SVG
soffice --headless --convert-to svg --outdir /tmp docs/architecture/effiped-architecture.pptx
python tools/architecture/clean_svg.py /tmp/effiped-architecture.svg docs/architecture/effiped-architecture.svg
```

Then `python tools/validate_release.py`.

## Why `clean_svg.py` exists

LibreOffice's SVG export is built for its slideshow player, and three of its
habits break a static diagram:

1. It bundles a ~450 KB touch-gesture script — dead weight here, and blocked by
   the site's `script-src 'self'` policy anyway.
2. **It wraps the real slide in `<g visibility="hidden">` and relies on that
   script to reveal it.** Removing the script without unhiding the wrapper ships
   an SVG that renders completely blank. The script raises an error rather than
   emitting a silently invisible file if that wrapper ever stops matching.
3. It emits a viewBox in 1/100 mm with no `width`/`height`, leaving the image
   with no intrinsic size — Chrome then rasterises it as blank in `<img>` and
   `<canvas>`.

It also adds the `<title>`/`<desc>` pair that `tools/validate_release.py`
requires for accessibility.

## Verifying a rebuild

Rasterise the SVG and the PNG at the same size and compare. A correct export
lands within a mean absolute difference of roughly 15/255 — that residual is
text antialiasing between the two rasterisers. A blank SVG shows up immediately
as a near-total difference against the dark background.
