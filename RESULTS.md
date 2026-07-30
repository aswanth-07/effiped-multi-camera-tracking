# Results and Protocol Ledger

The website, README, report, charts, and tests read the same checked-in fixture:
`research/results/summary.json`.

## EffiPed system benchmarks

| Evaluation | Metric | Value |
|---|---|---:|
| P-DESTRE fold-0 validation, cross-camera Protocol D | Rank-1 | 62.8% |
| P-DESTRE fold-0 validation | detection mAP@0.5 | 90.74% |
| P-DESTRE fold-0 test, cross-camera Protocol E | Rank-1 | 61.3% |
| P-DESTRE fold-0 test | detection mAP@0.5 | 88.4% |
| MOT17 val-half Protocol A | MOTA | 64.08 |
| MOT17 val-half Protocol A | IDF1 | 74.24 |
| MOT17 val-half Protocol A | HOTA | 61.34 |

EffiPed Tier-1 contains 7.78M parameters and runs at approximately 18 FPS for the complete
tracking pipeline at 1088×608 on an NVIDIA RTX 5080 Laptop GPU.

The browser replay is not used to measure these numbers. It is an interactive presentation
of archived detector, tracker, and cross-camera association output.

## Descriptor research extensions

| Study | Boundary | Rank-1 gain | mAP gain |
|---|---|---:|---:|
| PartJDE | matched fold-0 validation readout | +6.66 pp | — |
| BoxJDE | five-fold source-level detected / E2E | +13.64 pp | +12.94 pp |
| BoxJDE | five-fold natural predicted boxes | +13.31 pp | +12.29 pp |
| BoxJDE | five-fold natural end-to-end | +13.01 pp | +12.00 pp |

PartJDE's separate evaluation reports 7.92M parameters and 27.0 FPS. BoxJDE uses a
constructed P-DESTRE per-date descriptor-readout ablation, not official Task 4; its full
evidence and technical report live in the
[BoxJDE repository](https://github.com/aswanth-07/boxjde-person-search).

## Interpretation boundary

Cross-camera similarities rank candidate evidence. They do not establish identity, and the
reported evaluations do not establish behavior for other sites, populations, cameras, or
operating conditions.
