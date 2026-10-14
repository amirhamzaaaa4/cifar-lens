FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN sed 's/^tensorflow==/tensorflow-cpu==/' requirements.txt > requirements-cpu.txt \
    && pip install --no-cache-dir -r requirements-cpu.txt \
    && useradd --create-home appuser
COPY . .
USER appuser
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
