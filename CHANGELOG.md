# Changelog

Notable changes to this project, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Dates are the dates of the commits that made each change.

## [Unreleased]

### Added

- An Architecture tab (key `5`): the system diagram beside the training objective,
  every detection and identity loss term with its weight, the training and inference
  settings, and the reported results. A unit test checks each printed value against
  `configs/system/effiped.yaml` and reads the results from `research/results/summary.json`
  (2026-10-14).
- `ARCHITECTURE.md`, a plain description of the model, the losses, training,
  inference, and how the repository fits together (2026-10-14).
- The architecture diagram is drawn by `tools/architecture/build_diagram.py` as a
  paper-style block diagram on the console's black ground, with a PNG export through
  the web app's Playwright Chromium. It now shows the compact two-scale trunk the
  configuration actually builds (stem and first two ConvNeXt V2 stages, stride 4 and 8)
  rather than four stages to stride 32 (2026-10-14).
- Every detection the detector returned on each processed frame, replayed from the
  same run as the person-search index (`tools/build_detection_fixture.py`), so the
  Detection workspace draws everyone in a frame rather than only the few stored views
  of indexed people, and the Tracking and Sources workspaces count every local track
  (2026-10-12).
- The Tracking workspace plays the source clip with every local track from the
  indexing run drawn over it, using the same ids as its table and the person index.
  Selecting a row jumps to where that track starts and emphasises it (2026-10-12).
- A Person Search UI walkthrough with source selection, ranked candidates, and
  full-frame review, plus attribution for the derived demonstration media (2026-10-05).

### Changed

- The console has five tabs: Person Search, Detection, Tracking, Sources, and
  Architecture. An EP mark, shorter labels, and removal of explanatory footnotes
  simplify the interface. Public documentation, model metadata, the configuration
  (`configs/system/effiped.yaml`, model key `effiped`), and the report use the EffiPed
  name consistently. The README preview image is a current screenshot of the console
  (2026-10-14).
- The default confidence floor is 0.35, the threshold the archived tracking results
  were run at; 0.45 hid the nearest pedestrians in camera 2. Drawn frames apply hard
  NMS over the decoder's soft-NMS output, seek to the middle of a frame so boxes land
  on the right image, and the strongest cross-camera links list each pair once
  (2026-10-12).
- Clip posters are the first frame of the clean source clip, not a frame from the
  archived tracked render, and the candidate caution quotes this session's archived
  pairwise precision instead of similarity figures from a different run. The Person
  Search walkthrough was re-recorded to match (2026-10-12).
- The README links directly to the hosted console and distinguishes its stored
  replay from the separate local inference API and archived benchmark results
  (2026-10-05).
- The contributor checklist includes the result, media, release, and browser
  verification commands (2026-10-05).
- The web interface is an identity review console: fixed chrome, one viewport, six
  workspaces on the number keys (Person Search, Detection, Tracking, Cross Camera,
  Sources, Model), thresholds in a panel that drops from the bar instead of a rail
  that spends a fifth of the window on controls touched once a session, and a status
  bar that prints what each threshold change cost. It replaces the six-panel demo
  workbench that mirrored the original PedestrianTracker layout (2026-09-18).
- Person search is a review workflow rather than a grid of results: one source clip
  with that camera's gallery beside it, a time window over the clip, and the ranked
  candidates for whoever is selected, each shown with the full frame it was cropped
  from and that frame's provenance (2026-09-18).
- Each threshold shows the distribution it is cutting, so a slider says how many
  detections, tracks or links survive the position it is in (2026-09-18).
- The viewport carries structure with distance and hairline rings instead of a border
  around every group. Content sits on a measure with a gutter that opens with the
  window, media stops at its native 960px rather than being upscaled into a
  full-window slab, and a camera's gallery reads as photographs of people rather than
  as a lattice of cards (2026-09-19).
- The six-panel demo workbench, which the console above replaced, had itself replaced
  the earlier single-pane demo console: Single Camera, Cross Camera, Person Search,
  Image Detection, Model Status and Research Context, mirroring the original
  PedestrianTracker application layout. Four P-DESTRE session `12-11-2019_3` clips
  ship pre-attached (2026-08-06).
- The architecture diagram was rebuilt from a clean deck; the SVG, PNG and editable
  PPTX under `docs/architecture/` are regenerated from that source (2026-08-06).
- Full-frame views for a single appearance are redrawn in the browser by seeking the
  shipped clip to the stored timestamp, so no per-appearance scene images are
  distributed (2026-07-31).

### Removed

- The Cross Camera and Model tabs, their unused styles and data, the separate
  descriptor-study configuration, and 23 media files the current demo no longer uses,
  including the Person Search walkthrough video, which showed the retired six-tab
  console. The PowerPoint diagram source and its LibreOffice export scripts are
  replaced by the SVG generator (2026-10-14).
- The archived per-camera tracked renders (8.4 MB), which the Tracking workspace no
  longer uses and whose track numbers matched nothing else in the console
  (2026-10-12).
- The GitHub Actions workflow. Lint, tests, the three release validators, the
  web typecheck, build, audit and end-to-end suite, and the secret scan are no
  longer run automatically on push. Every one of those commands is listed in
  [CONTRIBUTING.md](CONTRIBUTING.md) and still fails the same way when run.
- Dead code with no reachable caller: `EventOut` (declared but never applied, since
  the WebSocket route sends plain dicts), `SEBlock` (the fusion path uses BiFPN-style
  learned weights and referenced it nowhere), and five helpers in
  `effiped/runtime.py` left over from the Gradio-era UI that the FastAPI service
  replaced — `available_model_choices`, `preset_from_label`, `detection_count`,
  `iter_detection_rows` and `normalize_embedding_array`. No public behaviour changed.

### Fixed

- The build backend requires setuptools 77.0.3 or newer to support the declared
  SPDX license expression and license-file metadata (2026-10-05).
- The frontend test runner uses the patched Vitest release for
  GHSA-82fw-gwwq-j7x9 (2026-10-05).
- Browser tests start their own preview server on a dedicated port and fail if
  it is occupied, preventing checks from running against another application
  (2026-10-05).

- Two docstrings that no longer described the code: `runtime.default_presets` said
  "the two release artifacts" where the manifest declares one, and `model.py`
  described the stride-4 fusion as "ECA/SE" where `AdaptiveFeatureFusion` uses
  BiFPN-style fast-normalized learned weights.

## [1.0.0] — 2026-08-06

First public release: the packaged model, the local inference service, the hosted
demo, and the evidence and licensing record that governs what may be published.

### Added

- `effiped` package — the JDENet joint detection-and-embedding model (ConvNeXt V2
  backbone, stride-4 P2 fusion, CenterNet head, four-strip part-based 256-D
  descriptor), `UnifiedTracker` for camera-local association and
  `CrossCameraAssociator` for cross-view candidate ranking (2026-07-30).
- FastAPI identity-review service (`apps/api`) with the person-search job lifecycle,
  WebSocket progress stream and asset store; four console entry points
  `effiped-app`, `effiped-train`, `effiped-eval` and `effiped-demo` (2026-07-30).
- React/Vite demo workbench (`apps/web`), deployed as a static, precomputed replay
  (2026-07-30).
- A single source of truth for published evidence, `research/results/summary.json`,
  read by the README, `RESULTS.md`, `MODEL_CARD.md`, the website and the generated
  technical report (2026-07-30).
- Quality gates, run from `CONTRIBUTING.md`: `tools/validate_results.py` pins every published metric and
  the claim-boundary language, `tools/validate_media.py` verifies attribution and
  SHA-256 for every P-DESTRE-derived asset, and `tools/validate_release.py` enforces
  repository hygiene (2026-07-30).
- Release governance: `LICENSE` (Apache-2.0), `DATA_LICENSES.md` recording the
  checkpoint-publication HOLD, `MODEL_CARD.md`, `SECURITY.md`, `CITATION.cff` and
  `THIRD_PARTY_NOTICES.md` (2026-07-30).
- Versioned model manifest (`src/effiped/model_manifest.json`) describing
  `effiped-v1.pt`, and a checksum-verified download path (2026-07-30).

### Known limitations at this release

- **No model weights are published.** Training-data redistribution terms are
  unresolved across P-DESTRE, MOT17/MOT20, SOMPT22 and CrowdHuman, so
  `weights_status` is `withheld_pending_dataset_rights_review` and every model
  reports `available: false`. See [DATA_LICENSES.md](DATA_LICENSES.md).
- The hosted demo performs no inference; it replays archived output.
- See [Limitations](README.md#limitations) for the full list.

[Unreleased]: https://github.com/aswanth-07/effiped-multi-camera-tracking/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aswanth-07/effiped-multi-camera-tracking/releases/tag/v1.0.0
