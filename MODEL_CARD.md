# EffiPed model card

EffiPed detects pedestrians and describes their appearance using one ConvNeXt V2 backbone (7.78M parameters). Its detection branch uses CenterNet outputs. Its appearance branch uses RoIAlign, four body strips, Coordinate Attention, and a normalized 256-dimensional descriptor. A BoT-SORT-style tracker follows people within a camera. [ARCHITECTURE.md](ARCHITECTURE.md) gives the layers, losses and training settings.

## Intended use

- Non-commercial research and education.
- Study of the documented model and evaluation methods.
- Human review of camera video the user is authorized to process.
- A portfolio demonstration using saved, attributed results.

## Uses outside this scope

- Automated identity decisions or claims that two observations prove identity.
- Face recognition or biometric identification.
- Surveillance without a specific assessment of legality, privacy, consent, security, retention, bias, and human review.
- Commercial use while training-data terms remain non-commercial or unresolved.

## Evidence

Archived measurements report 62.8% validation and 61.3% test cross-camera Rank-1 on P-DESTRE, plus 64.08 MOTA, 74.24 IDF1, and 61.34 HOTA on MOT17 validation half. See [RESULTS.md](RESULTS.md) for the conditions and evidence limits.

The browser demo uses a research version with full-person descriptors. Its saved output is separate from the part-based model measurements above. The local API ranks cross-camera candidates by cosine similarity; the library's bounded gallery policy is not connected to that API.

## Limitations

Occlusion, inaccurate boxes, pose, similar clothing, lighting, camera layout, time gaps, crowd density, and changes in the video source can affect scores. A high similarity is a candidate for review, not proof of identity. Scores are not calibrated probabilities.

No fairness analysis, repeated-run uncertainty, or study of reviewer accuracy is included. Evaluation on the recorded datasets does not establish performance elsewhere. Public weights and raw evaluation records are unavailable, which prevents independent reproduction from this repository.

## Model weights

The manifest names `effiped-v1.pt`. Public distribution is on hold pending review of training-data redistribution rights. Loading an authorized checkpoint uses `pretrained=false` to avoid downloading a separate backbone.

Maintainer: [Aswanth Raj](https://github.com/aswanth-07).
