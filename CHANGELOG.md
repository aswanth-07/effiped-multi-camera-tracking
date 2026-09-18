# Changelog

Notable changes to this project, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Dates are the dates of the commits that made each change.

## [Unreleased]

### Changed

- The demo workbench replaces the earlier single-pane demo console with six panels:
  Single Camera, Cross Camera, Person Search, Image Detection, Model Status and
  Research Context, mirroring the original PedestrianTracker application layout.
  Four P-DESTRE session `12-11-2019_3` clips ship pre-attached (2026-08-06).
- The architecture diagram was rebuilt from a clean deck; the SVG, PNG and editable
  PPTX under `docs/architecture/` are regenerated from that source (2026-08-06).
- Full-frame views for a single appearance are redrawn in the browser by seeking the
  shipped clip to the stored timestamp, so no per-appearance scene images are
  distributed (2026-07-31).

### Removed

- Dead code with no reachable caller: `EventOut` (declared but never applied, since
  the WebSocket route sends plain dicts), `SEBlock` (the fusion path uses BiFPN-style
  learned weights and referenced it nowhere), and five helpers in
  `effiped/runtime.py` left over from the Gradio-era UI that the FastAPI service
  replaced — `available_model_choices`, `preset_from_label`, `detection_count`,
  `iter_detection_rows` and `normalize_embedding_array`. No public behaviour changed.

### Fixed

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
- Quality gates run in CI: `tools/validate_results.py` pins every published metric and
  the claim-boundary language, `tools/validate_media.py` verifies attribution and
  SHA-256 for every P-DESTRE-derived asset, and `tools/validate_release.py` enforces
  repository hygiene (2026-07-30).
- Release governance: `LICENSE` (Apache-2.0), `DATA_LICENSES.md` recording the
  checkpoint-publication HOLD, `MODEL_CARD.md`, `SECURITY.md`, `CITATION.cff` and
  `THIRD_PARTY_NOTICES.md` (2026-07-30).
- Versioned model manifest (`src/effiped/model_manifest.json`) describing
  `effiped-tier1-v1.pt`, and a checksum-verified download path (2026-07-30).

### Known limitations at this release

- **No model weights are published.** Training-data redistribution terms are
  unresolved across P-DESTRE, MOT17/MOT20, SOMPT22 and CrowdHuman, so
  `weights_status` is `withheld_pending_dataset_rights_review` and every model
  reports `available: false`. See [DATA_LICENSES.md](DATA_LICENSES.md).
- The hosted demo performs no inference; it replays archived output.
- See [Limitations](README.md#limitations) for the full list.

[Unreleased]: https://github.com/aswanth-07/effiped-multi-camera-tracking/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aswanth-07/effiped-multi-camera-tracking/releases/tag/v1.0.0
