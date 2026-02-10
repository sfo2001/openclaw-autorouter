FROM docker.litellm.ai/berriai/litellm:main-stable

# Copy LiteLLM config and autorouter source
COPY config.yaml.example /app/config.yaml
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Phase 2: FastAPI on 4000 (external), LiteLLM on 4001 (internal)
EXPOSE 4000

ENTRYPOINT ["/app/entrypoint.sh"]
