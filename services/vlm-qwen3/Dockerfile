# Reuses the same pinned SGLang build as llm-huihui-qwen: v0.5.19 is well past
# v0.5.3, the release that added Qwen3-VL family support, and the digest pins
# the multi-arch image.
FROM lmsysorg/sglang:v0.5.19-cu129-runtime@sha256:fe08f6f52bea380415792849a703d92279c86320f8c39283bcd04043f8b63c41

# This service decodes images/video for a vision-language model, so it hits
# the same TorchCodec/FFmpeg gap as llm-huihui-qwen: the pinned runtime ships
# TorchCodec but not the FFmpeg shared libraries it probes. Install FFmpeg and
# fail the build if TorchCodec still cannot load its native extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && python3 -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec FFmpeg backend loaded')" \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /opt/private-model-hosting/entrypoint.sh
RUN chmod 0555 /opt/private-model-hosting/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/opt/private-model-hosting/entrypoint.sh"]
