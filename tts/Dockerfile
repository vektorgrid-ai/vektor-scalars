FROM workers-common
ENV PORT=8000

WORKDIR /opt/assistant

RUN pip install --upgrade pip

# Optimize layer caching so that dependencies are only reinstalled if pyproject.toml changes
COPY ./pyproject.toml ./tts/
RUN pip install --no-cache-dir ./tts/
COPY ./ ./tts/

EXPOSE $PORT

CMD ["python", "-m", "tts.worker"]
