@echo off
SETLOCAL

REM --- 설정 ---
SET IMAGE_NAME=yuneu6112/kseb-ai
SET DEFAULT_TAG=latest
REM --- 설정 끝 ---

REM 버전 태그 설정 (첫 번째 인자가 없으면 'latest' 사용)
SET TAG=%1
IF "%TAG%"=="" (
    SET TAG=%DEFAULT_TAG%
)

ECHO.
ECHO ==================================================
ECHO  Building Docker Image: %IMAGE_NAME%:%TAG%
ECHO ==================================================
ECHO.

REM Docker 이미지 빌드
docker build -t %IMAGE_NAME%:%TAG% .
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Docker build failed.
    GOTO :EOF
)

ECHO.
ECHO Build successful.
ECHO.

REM 'latest'가 아닌 다른 버전으로 빌드했다면, 'latest' 태그도 추가
IF /I NOT "%TAG%"=="%DEFAULT_TAG%" (
    ECHO Tagging %IMAGE_NAME%:%TAG% as %IMAGE_NAME%:%DEFAULT_TAG%...
    docker tag %IMAGE_NAME%:%TAG% %IMAGE_NAME%:%DEFAULT_TAG%
)

ECHO ==================================================
ECHO  Pushing to Docker Hub...
ECHO ==================================================
ECHO.

REM Docker Hub에 푸시 (버전 태그)
docker push %IMAGE_NAME%:%TAG%
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Failed to push %IMAGE_NAME%:%TAG%.
    GOTO :EOF
)

REM 'latest' 태그도 푸시
IF /I NOT "%TAG%"=="%DEFAULT_TAG%" (
    docker push %IMAGE_NAME%:%DEFAULT_TAG%
)

ECHO.
ECHO ==================================================
ECHO  Process completed successfully!
ECHO ==================================================

:EOF
ENDLOCAL
