# Contributing

Keep contributions focused on the contest system, local identity-review workflow, or verified
research evolution. Add tests proportional to the change and run:

```bash
ruff check .
pytest
cd apps/web
npm ci
npm run typecheck
npm run test
npm run build
npm audit --audit-level=high
```

Do not commit datasets, checkpoints, source videos, runtime crops, secrets, absolute local
paths, unrelated projects, or unsupported claims. Every P-DESTRE-derived asset must be
non-commercial, attributed, hashed, and listed in `docs/media/ASSET_MANIFEST.json`.

Code contributions are Apache-2.0. Media and third-party material retain their own compatible
terms.
