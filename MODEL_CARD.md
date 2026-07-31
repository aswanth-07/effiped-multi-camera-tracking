# EffiPed Model Card

## Summary

EffiPed Tier-1 is a compact joint detection and embedding model built around ConvNeXt V2,
P2/P3 fusion, CenterNet outputs, RoIAlign, four body strips, Coordinate Attention, and a
256-D normalized descriptor. BoT-SORT handles camera-local temporal association; a bounded
gallery ranks cross-camera candidates.

## Intended use

- Non-commercial research and education.
- Reproduction of the documented system and matched descriptor evaluations.
- Local, human-in-the-loop review of user-authorized camera video.
- Portfolio demonstration using precomputed, attributed media.

## Out-of-scope use

- Automated identity decisions or claims of identity.
- Face recognition or biometric identification.
- Public or covert surveillance without a purpose-specific legal, privacy, consent,
  security, retention, bias, and human-review assessment.
- Commercial use while training-data terms remain non-commercial or unresolved.

## Evidence boundary

Published system evidence includes 62.8% validation and 61.3% test cross-camera Rank-1 on
P-DESTRE, plus 64.08 MOTA, 74.24 IDF1, and 61.34 HOTA on MOT17 val-half. See
`RESULTS.md` for the complete protocol ledger.

## Limitations

Scores are sensitive to occlusion, detector localization, pose, clothing ambiguity,
illumination, camera calibration, time gaps, domain shift, and crowd density. A high cosine
similarity is candidate evidence, not proof that two observations depict the same person.
The published evaluation does not establish performance for other locations, populations,
or camera networks.

## Artifact status

The versioned manifest describes `effiped-tier1-v1.pt`, but public checkpoint distribution
is on hold. Runtime construction always uses `pretrained=false` when loading a checkpoint,
avoiding an unnecessary backbone download.

Maintainer: [Aswanth Raj](https://github.com/aswanth-07)
