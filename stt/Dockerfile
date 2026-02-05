FROM workers-common
ENV PORT=8000

WORKDIR /opt/assistant

RUN pip install --upgrade pip

# Optimize layer caching so that dependencies are only reinstalled if pyproject.toml changes
COPY ./pyproject.toml ./stt/
RUN pip install --no-cache-dir ./stt/
COPY ./ ./stt/

EXPOSE $PORT

CMD ["python", "-m", "stt.worker"]
