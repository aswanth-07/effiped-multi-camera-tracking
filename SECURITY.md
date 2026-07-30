# Security Policy

Report vulnerabilities through GitHub private vulnerability reporting for
`aswanth-07/effiped-multi-camera-tracking`; do not publish private video, identity data,
tokens, or workstation paths in an issue.

- Keep `effiped-app` bound to loopback or authenticated infrastructure.
- Use a narrow `EFFIPED_ALLOWED_ORIGINS` allowlist.
- Set a conservative `EFFIPED_MAX_UPLOAD_MB`.
- Store `EFFIPED_RUNTIME_DIR` on protected local storage.
- Delete jobs after review; the endpoint removes uploads and generated assets.
- Treat video, crops, descriptors, and candidate rankings as sensitive personal data.
- Verify any privately supplied checkpoint before loading it.
