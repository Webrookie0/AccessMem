FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
ARG INSTALL_EVAL=false

COPY requirements*.txt requirements.lock ./
RUN if [ "$INSTALL_EVAL" = "true" ]; then \
        pip install -r requirements-eval.txt; \
    else \
        pip install -r requirements.txt; \
    fi

RUN useradd --create-home --uid 10001 accessmem \
    && mkdir -p /data \
    && chown accessmem:accessmem /data
COPY --chown=accessmem:accessmem . .
USER accessmem
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
