# License and launch gates

Last checked: 2026-11-03. This is engineering tracking, not legal advice.

## Huihui Qwen3.8 27B abliterated

- The model repository declares Apache-2.0.
- Engineering status: permitted by the declared model license for commercial
  serving, subject to product safety and applicable law.
- Source: https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated

## Qwen Image 2.1

- License: Qwen Research License, not Apache-2.0.
- The license limits the granted rights to non-commercial research/evaluation
  unless a separate commercial license is obtained from Qwen.
- Status: **blocked for paid/commercial traffic**. Contact
  `model-business@notice.qwencloud.com` and retain the signed approval before
  enabling it in a paid privacy tier.
- Sources:
  - https://huggingface.co/Qwen/Qwen-Image-2.1
  - https://github.com/QwenLM/Qwen-Image-2.1

## LUSTIFY v10 / Krea 2

This checkpoint is a derivative of Krea 2 and has two independent gates.

### Krea base license

- Commercial use is allowed under the community license only while total
  company-wide trailing-12-month revenue is below USD 1,000,000. At or above
  that threshold an enterprise license is required.
- Section 4.2 requires reasonable and appropriate content filters. The AUP
  prohibits CSAM, NCII, rights violations, unlawful explicit content, removal
  of safety measures, and other listed misuse.
- Sources:
  - https://www.krea.ai/krea-2-licensing
  - https://www.krea.ai/krea-2-use-policy

### LUSTIFY uploader permissions

- Civitai API metadata for model `573152` currently reports commercial-use
  flags `RentCivit` and `Image`, while derivatives and alternate licensing are
  disabled. That does not clearly permit operating a general off-platform paid
  inference API.
- Status: **blocked for paid/public API traffic** until the checkpoint author
  grants explicit written permission and counsel confirms both license layers.
- The service remains in Compose's explicit `restricted` profile for internal
  evaluation only.
- Source: https://civitai.com/models/573152

## Launch checklist

- [ ] Signed Qwen Image commercial license retained.
- [ ] Written LUSTIFY author permission for the intended API/business model.
- [ ] Krea enterprise license if the revenue threshold applies.
- [ ] Counsel confirms all checkpoint/base-model terms and jurisdictions.
- [ ] Required attribution and notices shipped.
- [ ] LUSTIFY safety controls, age assurance, consent/NCII response, abuse
      reporting, and moderation have passed launch review.
