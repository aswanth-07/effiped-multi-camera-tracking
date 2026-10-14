# Reported results

The archived measurements below come from `research/results/summary.json`. The report and result validator use that same file. The browser demo replays saved output from a research version of EffiPed and does not measure these scores.

| Evaluation | Metric | Value |
|---|---|---:|
| P-DESTRE validation, retrieval across cameras | Rank-1 | 62.8% |
| P-DESTRE validation | Detection mAP@0.5 | 90.74% |
| P-DESTRE test, retrieval across cameras | Rank-1 | 61.3% |
| P-DESTRE test | Detection mAP@0.5 | 88.4% |
| MOT17 validation half | MOTA | 64.08 |
| MOT17 validation half | IDF1 | 74.24 |
| MOT17 validation half | HOTA | 61.34 |

Rank-1 is the share of searches where the correct person appears first. Detection mAP@0.5 measures detection quality at a box-overlap threshold of 0.5. MOTA combines missed detections, false detections, and identity switches; IDF1 measures identity consistency; HOTA combines detection and association quality.

The MOT17 measurement used minimum visibility 0.3, detection threshold 0.30, suppression threshold 0.60, and BoT-SORT on sequences held out of training. EffiPed contains 7.78M parameters. The complete tracking pipeline ran at approximately 18 frames per second at 1088 × 608 on an NVIDIA RTX 5080 Laptop GPU.

## What the evidence supports

These are single archived measurements on the recorded dataset splits. No repeated-run variation, confidence intervals, seed policy, or significance tests are reported. The weights and raw evaluation records are unavailable, so the numbers cannot be reproduced from this repository alone.

The results do not establish performance for other sites, populations, cameras, or operating conditions. Cross-camera similarities rank candidates for human review and do not establish identity.

The results file also preserves an earlier four-camera session diagnostic: 150 frames per clip, 367 local tracks, 45 cross-camera groups, and pairwise association precision 0.77. Those totals describe that archived run; they are separate from the filtered people and matches shown in the current browser demo.
