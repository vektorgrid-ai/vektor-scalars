FROM workers-common
ENV PORT=8000
ENV OLLAMA_MODELS=/ollama

RUN mkdir -p ${OLLAMA_MODELS}

RUN apt-get update && apt-get install -y curl && apt-get install -y zstd
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /opt/assistant

RUN pip install --upgrade pip

# Optimize layer caching so that dependencies are only reinstalled if pyproject.toml changes
COPY ./pyproject.toml ./router/
RUN pip install --no-cache-dir ./router/
COPY ./ ./router/

EXPOSE $PORT

CMD ["sh", "-c", "ollama serve & exec python -m router.worker"]
