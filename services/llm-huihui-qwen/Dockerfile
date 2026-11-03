# SGLang 0.5.19 is the first stable release with Qwen3.8-27B support and the
# final release published for CUDA 12.9. The digest pins the multi-arch image.
FROM lmsysorg/sglang:v0.5.19-cu129-runtime@sha256:fe08f6f52bea380415792849a703d92279c86320f8c39283bcd04043f8b63c41

# The pinned runtime contains TorchCodec but not the FFmpeg shared libraries it
# probes while SGLang imports its multimodal processors. Install the distro's
# supported FFmpeg build and fail the image build if TorchCodec still cannot
# load its native extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && python3 -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec FFmpeg backend loaded')" \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /opt/private-model-hosting/entrypoint.sh
RUN chmod 0555 /opt/private-model-hosting/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/opt/private-model-hosting/entrypoint.sh"]
