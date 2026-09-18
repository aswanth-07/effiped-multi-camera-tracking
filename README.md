**EffiPed detects pedestrians across several fixed cameras and ranks which observations
plausibly show the same person, so that a human can review them.** For engineers and
researchers evaluating compact multi-camera re-identification — and for anyone who wants to
try that workflow in a browser, with nothing to install.

<div align="center">
  <img src="docs/social-preview.png" alt="EffiPed multi-camera pedestrian tracking and identity-review system" width="100%">

  # EffiPed

  ## Multi-Camera Pedestrian Detection, Tracking & Re-Identification using Joint ConvNeXt V2 Architecture

  By [Aswanth Raj](https://github.com/aswanth-07)

  [![Software: Apache-2.0](https://img.shields.io/badge/software-Apache--2.0-22c7b8)](LICENSE)
  [![Media: CC BY-NC-SA 4.0](https://img.shields.io/badge/P--DESTRE_media-CC_BY--NC--SA_4.0-70b8ff)](docs/media/LICENSE.md)
</div>

EffiPed is a compact video-intelligence system that detects pedestrians, maintains camera-local
tracks, and ranks cross-camera identity candidates for human review. Its React
investigation console is available as a precomputed browser demo; the same workflow can
connect to local FastAPI/CUDA inference when an authorized checkpoint is available.

> [!IMPORTANT]
> Ranked matches are reviewable appearance evidence, not proof of identity. The hosted
> experience is a precomputed, non-commercial research demonstration. Public model weights
> remain withheld while training-data redistribution terms are unresolved.

## Try the identity review console

The hosted UI is an application console: a fixed top bar, one viewport, a thresholds
panel that drops from the bar when it is asked for, and a status bar that reports the
job. Four P-DESTRE session `12-11-2019_3` clips are already attached, as though you had
uploaded them. Six workspaces, switchable with the number keys:

- **Person Search** (`1`), the workspace the console exists for. Pick a camera, pick
  anyone the pipeline found in it, and review the ranked candidates from the other
  cameras beside the full frame each one was cropped from.
- **Detection** (`2`), one frame with its stored detections redrawn at the current
  confidence and box-height floors.
- **Tracking** (`3`), a clip's tracked render beside a live table of the tracks in it.
- **Cross Camera** (`4`), the camera-pair link matrix, the strongest links at the
  current threshold, and the archived per-pair precision diagnostic.
- **Sources** (`5`), the attached clips and their specifications.
- **Model** (`6`), the loaded artifact, the settings this replay's index was built
  with, the benchmarks by protocol, and the boundaries.

A threshold moves the whole job rather than one panel: raising the confidence floor
drops detections, which drops tracks, which drops links, and the status bar prints what
each change cost.

```bash
cd apps/web
npm install
npm run dev
```

The controls are live, but the hosted build performs no inference: each **Run** replays a
precomputed result. Full-frame views are redrawn in the browser by seeking the shipped
clip to the appearance's timestamp and stroking its stored box, so no per-appearance
scene images ship.

No synthetic browser boxes are drawn over the footage. The boxes baked into the replay
videos are the annotations rendered by the original tracking pipeline.

> [!NOTE]
> The person-search index was computed offline with the BoxJDE research checkpoint,
> because the EffiPed Tier-1 weights are withheld pending dataset rights review. The
> Person Search and Model Status panels both state this. Regenerate it with
> [`tools/build_person_search_fixture.py`](tools/build_person_search_fixture.py).

## System

One ConvNeXt V2 feature hierarchy supports CenterNet-style detection and a 256-D
part-aware identity descriptor. RoIAlign extracts a person feature map, four horizontal
body strips retain local appearance, and Coordinate Attention fuses the visible evidence.
BoT-SORT combines motion, overlap, and appearance for local temporal association; the
gallery then ranks possible cross-camera matches for an analyst.

| Evaluation | Result |
|---|---:|
| P-DESTRE validation cross-camera Rank-1 | **62.8%** |
| P-DESTRE test cross-camera Rank-1 | **61.3%** |
| P-DESTRE validation / test detection mAP@0.5 | **90.74% / 88.4%** |
| MOT17 val-half MOTA / IDF1 / HOTA | **64.08 / 74.24 / 61.34** |
| EffiPed Tier-1 footprint | **7.78M · ≈18 full-pipeline FPS** |

Each value has a protocol label in [RESULTS.md](RESULTS.md). The interactive replay is an
application demonstration, not a benchmark run.

## Architecture

[![EffiPed end-to-end architecture](docs/architecture/effiped-architecture.svg)](docs/architecture/effiped-architecture.svg)

The diagram is also available as an
[editable PowerPoint](docs/architecture/effiped-architecture.pptx).

## Repository map

```text
src/effiped/          installable model, descriptors, tracking, runtime
apps/api/             FastAPI local-GPU service and job lifecycle
apps/web/             React/Vite demo workbench UI and hosted replay
configs/system/       active EffiPed and matched PartJDE configurations
research/results/     single source of truth for published evidence
research/report/      generated technical report
docs/architecture/    editable diagram source and web exports
docs/media/           optimized, attributed demonstration media
tools/ and tests/     validation, regression, and release checks
```

## Run live inference locally

Python 3.11 and an NVIDIA GPU are recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[runtime]"
effiped-app
```

Place an authorized checkpoint in `EFFIPED_WEIGHTS_DIR`. When none is present, the API
reports the model as unavailable without exposing a local filesystem path.

```bash
effiped-train --config configs/system/effiped-tier1.yaml
effiped-eval --config configs/system/effiped-tier1.yaml
effiped-demo
```

| Variable | Purpose |
|---|---|
| `EFFIPED_WEIGHTS_DIR` | authorized local model artifacts |
| `EFFIPED_RUNTIME_DIR` | temporary uploads, crops, and job assets |
| `EFFIPED_DEVICE` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `EFFIPED_MAX_UPLOAD_MB` | per-video upload limit |
| `EFFIPED_ALLOWED_ORIGINS` | comma-separated CORS allowlist |

## Public API

- `GET /api/health`
- `GET /api/models`
- `POST /api/person-search/jobs`
- `GET /api/person-search/jobs/{job_id}` and `/stream`
- `GET .../people`, `/detections`, `/tracks`, and `/matches`
- `POST .../search-by-example`
- `DELETE /api/person-search/jobs/{job_id}`
- `GET /api/assets/{asset_id}`

Deleting a job removes uploaded video and generated assets.

## Limitations

Written from what this repository can and cannot show, not from modesty.

- **A ranked match is not an identification.** Cross-camera similarity orders candidate
  appearance evidence for a person to review. The demo fixture makes the reason visible:
  non-matches score 0.997–0.998 against matches at 0.998–0.999, so the *ranking* is useful
  and the absolute score is not. There is no calibration and no decision threshold.
- **The shipped service ranks cross-camera candidates by cosine similarity alone.**
  `PersonSearchManager.matches` in
  [`apps/api/person_search_service.py`](apps/api/person_search_service.py) compares the
  query descriptor against every other person, drops anyone from the query's own clip,
  sorts, and truncates. The `cross_camera:` policy in
  [`configs/system/effiped-tier1.yaml`](configs/system/effiped-tier1.yaml), which sets a
  match threshold, gallery size, temporal window, transition weight and a global identity
  cap, belongs to `CrossCameraAssociator`, and nothing outside `tests/test_tracker.py`
  reaches that class. The configured policy and the running behaviour are not the same
  thing, on the capability this project is named for.
- **No published weights, so nothing here reproduces the numbers.** Publication is on hold
  pending a dataset-rights review ([DATA_LICENSES.md](DATA_LICENSES.md)). `pip install`
  succeeds, `effiped-app` starts, and every model reports `available: false`. The reported
  results are attested by `research/results/summary.json` and defended against drift by
  `tools/validate_results.py`; they are not re-derivable from this repository alone.
- **Every published number is a single measurement on one fold.** No variance, interval,
  seed policy or significance test is reported for any value, and the FPS figure is
  approximate, from one device at one resolution.
- **Evaluated only on P-DESTRE fold 0 and MOT17 val-half.** Nothing here establishes
  behaviour for another site, population, camera network, or operating condition.
- **No fairness or subgroup analysis**, on a system that ranks people by appearance.
- **Nothing measures the review loop the system exists for.** There is no study of whether
  ranked candidates make a reviewer faster or more accurate.
- **The hosted demo performs no inference.** It replays archived output from the original
  PedestrianTracker application; the controls are live, but Run returns a stored result.
- **The local API has no authentication**, which is safe only because it binds `127.0.0.1`.
  The container image binds `0.0.0.0`, so publishing that port is a decision requiring its
  own review.
- **Two-thirds of the Python is neither linted nor tested.** `train.py`, `loss.py` and
  `dataset.py` are research code carried forward from the training workspace and are
  excluded from ruff; the inference path itself cannot be tested end to end without a
  checkpoint.

## Research connections

The later [BoxJDE Person Search](https://github.com/aswanth-07/boxjde-person-search)
repository isolates the full-person descriptor readout and documents its five-fold
P-DESTRE ablation. It is linked as related research; its code and report are not duplicated
here.

## Licensing and responsible use

Original software is © 2026 Aswanth Raj and licensed under Apache-2.0. P-DESTRE-derived
media under `docs/media/pdestre/` is separately licensed as a CC BY-NC-SA 4.0 adaptation
for this non-commercial demonstration. The
[asset manifest](docs/media/ASSET_MANIFEST.json) records the source, transformations,
hash, purpose, and license for every derived asset.

No dataset, source video, person-level benchmark record, checkpoint, or runtime crop is
included.

[Model card](MODEL_CARD.md) ·
[Data and weight-release audit](DATA_LICENSES.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md) ·
[P-DESTRE paper](https://arxiv.org/abs/2004.02782) ·
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
