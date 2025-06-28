# 1. Use an explicit Debian-based slim image
FROM python:3.9-slim-buster

# 2. Create & switch to a non-root user for better security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 3. Set working directory
WORKDIR /app

# 4. Streamline Python & pip behavior
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true

# 5. Install Python dependencies (leveraging Docker cache)
COPY requirements.txt .
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    pip install -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# 6. Copy application code and set proper ownership
COPY --chown=appuser:appgroup . .

# 7. Switch to the non-root user
USER appuser

# 8. Expose Streamlit’s default port
EXPOSE 8501

# 9. Entrypoint using bash so $PORT can be injected by platforms like Koyeb
ENTRYPOINT ["bash", "-c", "\
    streamlit run cpte_app3_experiment.py \
      --server.address=0.0.0.0 \
      --server.port ${PORT:-8501} \
"]
