FROM python:3.11-slim

WORKDIR /opt/assistant
RUN pip install --upgrade pip

# Optimize layer caching so that dependencies are only reinstalled if pyproject.toml changes
COPY ./pyproject.toml ./common/
RUN pip install --no-cache-dir ./common/
COPY ./ ./common/
