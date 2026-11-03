# Official vLLM-Omni release with native Qwen-Image-2.1 serving, step-wise
# scheduling, prefix-KV reuse, CUDA graphs, and bounded request admission.
FROM vllm/vllm-omni:v0.32.0@sha256:f713b909f16cf8c7901f17c3aee3f471cbee5fbde5603b7dbe2631630887d7c2

COPY entrypoint.sh /opt/private-model-hosting/entrypoint.sh
RUN chmod 0555 /opt/private-model-hosting/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/opt/private-model-hosting/entrypoint.sh"]
