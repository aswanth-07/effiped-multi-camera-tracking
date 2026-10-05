# Contributing

Keep contributions focused on the EffiPed model, local identity-review workflow, or verified
descriptor research. Add tests proportional to the change and run:

```bash
ruff check .
pytest
python tools/validate_results.py --config configs/system/effiped-tier1.yaml
python tools/validate_media.py
python tools/validate_release.py
cd apps/web
npm ci
npm run typecheck
npm run test
npm run build
npm run test:e2e
npm audit --audit-level=high
```

Browser tests start their own preview server on port 4187. Set `EFFIPED_E2E_PORT`
to another free port if needed; the suite fails if that port is already occupied.

Do not commit datasets, checkpoints, source videos, runtime crops, secrets, absolute local
paths, unrelated projects, or unsupported claims. Every P-DESTRE-derived asset must be
non-commercial, attributed, hashed, and listed in `docs/media/ASSET_MANIFEST.json`.

Code contributions are Apache-2.0. Media and third-party material retain their own
compatible terms.
