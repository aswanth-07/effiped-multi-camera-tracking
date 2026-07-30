# Results and Protocol Ledger

All website cards, charts, README values, and report tables are generated from
`research/results/summary.json`.

## 1. Verified contest system

| Evaluation | Metric | Value |
|---|---|---:|
| P-DESTRE fold-0 validation, cross-camera Protocol D | Rank-1 | 62.8% |
| P-DESTRE fold-0 validation | detection mAP@0.5 | 90.74% |
| P-DESTRE fold-0 test, cross-camera Protocol E | Rank-1 | 61.3% |
| P-DESTRE fold-0 test | detection mAP@0.5 | 88.4% |
| MOT17 val-half Protocol A | MOTA | 64.08 |
| MOT17 val-half Protocol A | IDF1 | 74.24 |
| MOT17 val-half Protocol A | HOTA | 61.34 |

Canonical Tier-1 footprint: 7.78M parameters and approximately 18 FPS for the full tracking
pipeline at 1088×608 on an NVIDIA RTX 5080 Laptop GPU.

## 2. Contest submission snapshot

The submitted poster displayed 7.92M parameters, 22 FPS, and 62.8% cross-camera Rank-1.
That snapshot is preserved as submitted. It is not substituted for the later canonical
Tier-1 measurement. The poster’s +16.2 percentage-point row combined multiple configuration
differences; it is not a pure part-readout ablation.

## 3. Post-contest evolution

| Study | Boundary | Rank-1 gain | mAP gain |
|---|---|---:|---:|
| PartJDE | matched fold-0 validation readout | +6.66 pp | — |
| BoxJDE | five-fold source-level detected / E2E | +13.64 pp | +12.94 pp |
| BoxJDE | five-fold natural predicted boxes | +13.31 pp | +12.29 pp |
| BoxJDE | five-fold natural end-to-end | +13.01 pp | +12.00 pp |

PartJDE’s separate evaluation reports 7.92M parameters and 27.0 FPS. BoxJDE’s primary
P-DESTRE protocol is a constructed per-date descriptor-readout ablation, not official Task 4.
