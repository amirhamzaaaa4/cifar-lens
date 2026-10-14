FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN sed 's/^tensorflow==/tensorflow-cpu==/' requirements.txt > requirements-cpu.txt \
    && pip install -r requirements-cpu.txt
RUN useradd --create-home --uid 1000 lens
COPY --chown=lens:lens . .
USER lens
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
# Optional: LENS_EXPERIMENT_URL=<release zip> downloads and verifies trained weights at startup.
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
