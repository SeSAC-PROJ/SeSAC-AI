# KSEB-AI Server

KSEB 프로젝트의 AI 파트 리포지토리입니다. FastAPI를 기반으로 동영상을 받아 시선, 감정 등을 분석하는 API 서버입니다.

## 🚀 주요 기술 스택

- **Web Framework**: FastAPI
- **Dependency Management**: Poetry
- **Deployment**: Docker

## ⚙️ 개발 환경 설정

이 프로젝트는 Poetry를 사용하여 의존성을 관리합니다.

1.  **Poetry 설치**: [공식 문서](https://python-poetry.org/docs/#installation)를 참고하여 시스템에 Poetry를 설치합니다.

2.  **의존성 설치**: 프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.
    ```bash
    poetry install
    ```

3.  **로컬 서버 실행**:
    ```bash
    poetry run uvicorn app.main:app --reload
    ```

## 🐳 Docker를 이용한 빌드 및 배포

이 프로젝트는 Docker 이미지를 빌드하여 배포합니다. 빌드 과정은 로컬에서 직접 수행합니다.

**전제 조건:**
- PC에 [Docker Desktop](https://www.docker.com/products/docker-desktop/)이 설치 및 실행되어 있어야 합니다.
- 터미널에서 `docker login` 명령어로 Docker Hub 계정에 로그인해야 합니다.

### 빌드 및 푸시 방법

프로젝트 루트 디렉토리에 있는 `build.bat` 스크립트를 사용합니다.

- **최신 버전으로 빌드 및 푸시:**
  ```bash
  build.bat
  ```

- **특정 버전 태그(예: `0.1.0`)로 빌드 및 푸시:**
  ```bash
  build.bat 0.1.0
  ```
  (이 경우, 지정된 버전 태그와 `latest` 태그가 모두 Docker Hub에 푸시됩니다.)

### 서버 실행

Docker Hub에 올라간 이미지를 사용하여 아래 명령어로 서버를 실행할 수 있습니다.
```bash
docker run -p 8000:8000 yuneu6112/kseb-ai:latest
```
