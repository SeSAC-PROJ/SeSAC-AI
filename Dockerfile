# ──── 1단계: 의존성 설치 ────
FROM python:3.10-slim AS builder
WORKDIR /app

# 1) 시스템 빌드 툴 + ffmpeg 포함
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      python3-dev \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 2) Poetry 설치 & system site-packages에 설치 설정
RUN pip install --no-cache-dir poetry \
 && poetry config virtualenvs.create false

# 3) 프로젝트 파일만 복사 → 의존성 설치 (캐시 활용)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

# ──── 2단계: 최종 실행 이미지 ────
FROM python:3.10-slim
WORKDIR /app

# 1) 시스템 런타임 툴만 설치
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 2) builder에서 설치된 Python 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 3) 소스 코드 복사
COPY app/ ./app

# 4) 포트 노출
EXPOSE 8000

# 5) 실행 커맨드
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]