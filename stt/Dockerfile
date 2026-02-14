FROM workers-common
ENV PORT=8000
ENV HF_HOME=/opt/assistant/.cache
ENV TRANSFORMERS_CACHE=/opt/assistant/.cache

# Allow choosing model at build time; defaults to "small". Keep as ARG so builds can vary model with --build-arg.
ARG WHISPER_MODEL=small
ENV WHISPER_MODEL=${WHISPER_MODEL}

WORKDIR /opt/assistant

RUN pip install --upgrade pip

COPY ./pyproject.toml ./stt/
RUN pip install --no-cache-dir ./stt/

# Pre-download the model into the cache during image build. Use CPU at build time so GPU is not required for image builds.
RUN python - <<'PY'
import os, sys
from faster_whisper import WhisperModel
model = os.environ.get('WHISPER_MODEL', 'small')
print(f'Pre-downloading faster-whisper model "{model}" (build-time uses CPU)')
try:
    # Build-time always uses CPU to avoid requiring GPUs in the build environment.
    WhisperModel(model, device='cpu', compute_type='int8')
    print('Model pre-download complete')
except Exception as e:
    print('Model pre-download failed:', e, file=sys.stderr)
    raise
PY

COPY ./ ./stt/

EXPOSE $PORT

CMD ["python", "-m", "stt.worker"]
