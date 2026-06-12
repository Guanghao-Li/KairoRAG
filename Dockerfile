FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY kairorag ./kairorag
COPY data/raw ./data/raw

RUN python -m pip install --upgrade pip \
    && python -m pip install .

CMD ["kairo", "--help"]
