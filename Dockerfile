# --- 1단계: 의존성 빌드 환경 ---
FROM python:3.10-slim as builder

# 시스템 의존성 설치: cmake는 dlib 빌드에 필요합니다.
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN pip install poetry

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 프로젝트 파일 복사
COPY poetry.lock pyproject.toml ./

# Poetry를 사용하여 의존성 설치 (개발용 라이브러리 제외)
# 이 때 라이브러리들은 Poetry의 기본 캐시 경로에 설치됩니다.
RUN poetry install --no-root --only main


# --- 2단계: 최종 실행 이미지 ---
FROM python:3.10-slim

# 시스템 의존성 설치: ffmpeg는 비디오/오디오 처리에 필요합니다.
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# builder 단계에서 설치된 Python 라이브러리들만 복사해옵니다.
# 이 과정 덕분에 최종 이미지에는 빌드에만 필요했던 cmake나 poetry가 포함되지 않아 가볍습니다.
COPY --from=builder /root/.cache/pypoetry/virtualenvs/kseb-ai-*-py3.10/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 애플리케이션 소스 코드 복사
COPY app/ ./app

# 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]