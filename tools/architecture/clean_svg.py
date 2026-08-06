"""Strip LibreOffice slideshow cruft from the exported SVG and make it accessible.

LibreOffice bundles a ~450 KB touch-gesture script for its slideshow player and
a pile of presentation metadata. None of it does anything for a static diagram,
and the site's CSP (`script-src 'self'`) would block the script regardless.
`tools/validate_release.py` additionally requires <title> and <desc> as direct
children of <svg>.

Edits are done as targeted text surgery rather than an ElementTree round-trip,
which would rewrite every namespace prefix.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

TITLE = "EffiPed system architecture"
DESC = (
    "Left-to-right pipeline. Four synchronised camera feeds are letterboxed to "
    "1088 by 608 and passed through one shared ConvNeXt V2 trunk, then an adaptive "
    "P2/P3 fusion neck producing 256 channels at stride 4. The fused features fork "
    "into two heads: a CenterNet detection head predicting heatmap, LTRB box, centre "
    "offset and IoU quality; and a part-aware identity head using RoIAlign over four "
    "horizontal body strips with Coordinate Attention, emitting a 256-D L2-normalised "
    "descriptor. BoT-SORT associates detections into camera-local tracks using Kalman "
    "motion, an IoU cascade and cosine appearance; a cross-camera gallery then ranks "
    "top-k candidates with mutual-visibility weighting. Both paths converge on analyst "
    "review, where a human confirms every match: candidates are appearance evidence, "
    "not proof of identity."
)


def clean(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    before = len(text)
    text, n = re.subn(r"<script\b[^>]*>.*?</script\s*>", "", text, flags=re.S)
    if n:
        notes.append(f"removed {n} <script> block(s), {before - len(text):,} bytes")

    # LibreOffice wraps the real slide in <g visibility="hidden"> and relies on
    # its slideshow script to reveal it. With the script gone the whole diagram
    # would render blank, so unhide the wrapper explicitly.
    text, n = re.subn(
        r'(<g class="SlideGroup">\s*)<g visibility="hidden">',
        r"\1<g>",
        text,
        count=1,
    )
    if n:
        notes.append("unhid the SlideGroup wrapper (script previously revealed it)")
    else:
        raise SystemExit(
            "expected a hidden <g> inside <g class=\"SlideGroup\">; LibreOffice's "
            "export shape changed — check whether the slide still renders."
        )

    # Slideshow-only metadata blocks LibreOffice emits.
    for tag in ("ooo:meta_slides", "ooo:slide_transitions"):
        text, n = re.subn(rf"<{tag}\b.*?</{tag}\s*>", "", text, flags=re.S)
        if n:
            notes.append(f"removed <{tag}>")

    # Drop an existing title/desc so re-runs stay idempotent.
    text = re.sub(r"<title\b[^>]*>.*?</title\s*>", "", text, count=1, flags=re.S)
    text = re.sub(r"<desc\b[^>]*>.*?</desc\s*>", "", text, count=1, flags=re.S)

    match = re.search(r"<svg\b[^>]*>", text)
    if not match:
        raise SystemExit("no <svg> root element found")
    inject = f"<title>{escape(TITLE)}</title><desc>{escape(DESC)}</desc>"
    text = text[: match.end()] + inject + text[match.end() :]
    notes.append("inserted <title> and <desc>")

    # LibreOffice emits a viewBox in 1/100 mm with no width/height, which leaves
    # the SVG with no intrinsic size: Chrome then rasterises it as a blank 0x0 in
    # <img> and canvas. Pin a 16:9 intrinsic size; CSS still scales it.
    root = re.search(r"<svg\b[^>]*>", text).group(0)
    patched = root
    if not re.search(r"\swidth=", patched):
        patched = patched[:-1] + ' width="1600" height="900">'
        notes.append("added intrinsic width/height (1600x900)")
    if "role=" not in patched:
        patched = patched[:-1] + ' role="img">'
        notes.append('added role="img"')
    if "aria-labelledby" not in patched:
        patched = patched[:-1] + ' aria-labelledby="effiped-title effiped-desc">'
        notes.append("added aria-labelledby")
    if patched != root:
        text = text.replace(root, patched, 1)

    # Tie the ids referenced by aria-labelledby to the nodes we injected.
    text = text.replace("<title>", '<title id="effiped-title">', 1)
    text = text.replace("<desc>", '<desc id="effiped-desc">', 1)

    return text, notes


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    original = src.read_text(encoding="utf-8")
    cleaned, notes = clean(original)
    dst.write_text(cleaned, encoding="utf-8")
    for note in notes:
        print(f"  {note}")
    print(f"  {len(original):,} -> {len(cleaned):,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
