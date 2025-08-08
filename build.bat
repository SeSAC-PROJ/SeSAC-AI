@echo off
SETLOCAL

CD /D "%~dp0"

SET IMAGE_NAME=yuneu6112/kseb-ai
SET DEFAULT_TAG=latest
SET CACHE_IMAGE_NAME=%IMAGE_NAME%:cache

SET TAG=%1
IF "%TAG%"=="" (
    SET TAG=%DEFAULT_TAG%
)

ECHO.
ECHO ==================================================
ECHO  Building Docker Image with Cache for: %IMAGE_NAME%:%TAG%
ECHO  This will create and push a cache to Docker Hub.
ECHO ==================================================
ECHO.

docker buildx build --platform linux/amd64 -t %IMAGE_NAME%:%TAG% --push --cache-to=type=registry,ref=%CACHE_IMAGE_NAME%,mode=max --cache-from=type=registry,ref=%CACHE_IMAGE_NAME% .

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Docker buildx failed.
    GOTO :EOF
)

ECHO.
ECHO ==================================================
ECHO  Process completed successfully!
ECHO ==================================================

:EOF
ENDLOCAL
