# License tracking

Keep this file accurate as the source of truth for what's legally cleared to
serve to paying customers vs. what's an accepted interim risk. Update it the
moment any of these statuses change.

## huihui-ai/Huihui-Qwen3.8-27B-abliterated

- **License**: Apache-2.0.
- **Status**: cleared for commercial use. No restrictions found.

## Qwen/Qwen-Image-2.1

- **License**: Qwen Research License Agreement (`license: other`,
  `license_name: qwen-research` on the HF model card).
- **Key clauses** (from the repo's own `LICENSE` file):
  - Section 1(i): "Non-Commercial" is defined as "for research or evaluation
    purposes only."
  - Section 2(a): rights are granted "FOR NON-COMMERCIAL PURPOSES ONLY."
  - Section 2(b): "You shall not use the Materials for any commercial purpose
    without obtaining a separate commercial license from us" — request one at
    `model-business@notice.qwencloud.com`.
  - Section 3 (Redistribution) is explicitly subject to Section 2 — bundling it
    into a paid product doesn't override the non-commercial restriction.
- **Status**: **ACCEPTED INTERIM RISK.** Decision made 2026-10-29/30: use it now
  for the Private tier, revisit before this tier bills real customers. This is a
  business/legal decision, not an engineering one — someone needs to actually
  email Qwen about a commercial license before this goes fully live for paying
  users.
- **Note**: the newer `Qwen-Image-2.1-Turbo` checkpoint was considered first and
  rejected for the same reason (same license, confirmed via its own LICENSE
  file) — Qwen-Image-2.1 (non-Turbo) doesn't fix this, it carries the identical
  restriction. The Apache-2.0-licensed alternative, if this risk becomes
  unacceptable later, is the one-generation-earlier pair `Qwen/Qwen-Image-2512`
  (T2I) + `Qwen/Qwen-Image-Edit-2511` (edit) — both confirmed Apache-2.0,
  smaller quality/speed trade-off (no 2.1-era improvements), same family.

## LUSTIFY V10 "Krea 2" (civitai.com/models/573152)

- **Base model license (Krea 2)**: community license with a **$1,000,000
  annual revenue cap** for free commercial use (per Krea's own enterprise
  license page and third-party HF mirrors). Above that revenue threshold, or
  for hosting an API endpoint to the Krea model, a paid enterprise license is
  required from Krea.
- **LUSTIFY V10's own version-specific license**: **OPEN — not yet confirmed.**
  Civitai's WebFetch access returned HTTP 403 during research; civarchive's
  mirror didn't show license/commercial-use terms for this specific version
  either. The Civitai "License" tab (id 2875936, historically CreativeML
  Open RAIL++-M with the uploader's own added use-restrictions) needs a manual,
  logged-in check on `civitai.com/models/573152` before this model is enabled
  for any paying user. **This is a hard gate, not a nice-to-have** — do this
  before step 3 of `ARCHITECTURE.md`'s Verification section goes live.
- **Status**: blocked on manual confirmation above. Treat as not cleared for
  commercial use until that check is done and logged here.
