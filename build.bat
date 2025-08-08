@echo off
SETLOCAL

SET IMAGE_NAME=yuneu6112/kseb-ai
SET DEFAULT_TAG=latest

SET TAG=%1
IF "%TAG%"=="" (
    SET TAG=%DEFAULT_TAG%
)

ECHO.
ECHO ==================================================
ECHO  Building Docker Image: %IMAGE_NAME%:%TAG%
ECHO ==================================================
ECHO.

docker build -t %IMAGE_NAME%:%TAG% .
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Docker build failed.
    GOTO :EOF
)

ECHO.
ECHO Build successful.
ECHO.

IF /I NOT "%TAG%"=="%DEFAULT_TAG%" (
    ECHO Tagging %IMAGE_NAME%:%TAG% as %IMAGE_NAME%:%DEFAULT_TAG%...
    docker tag %IMAGE_NAME%:%TAG% %IMAGE_NAME%:%DEFAULT_TAG%
)

ECHO ==================================================
ECHO  Pushing to Docker Hub...
ECHO ==================================================
ECHO.

docker push %IMAGE_NAME%:%TAG%
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Failed to push %IMAGE_NAME%:%TAG%.
    GOTO :EOF
)

IF /I NOT "%TAG%"=="%DEFAULT_TAG%" (
    docker push %IMAGE_NAME%:%DEFAULT_TAG%
)

ECHO.
ECHO ==================================================
ECHO  Process completed successfully!
ECHO ==================================================

:EOF
ENDLOCAL