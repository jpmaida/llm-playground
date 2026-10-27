@echo off
REM Windows. No macOS e no Linux o editor usa container-cli.
REM Escolhe docker quando o executavel existe e, senao, podman.

where docker >nul 2>&1
if not errorlevel 1 (
  docker %*
  exit /b
)

if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" (
  "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" %*
  exit /b
)

where podman >nul 2>&1
if not errorlevel 1 (
  podman %*
  exit /b
)

if exist "%ProgramFiles%\RedHat\Podman\podman.exe" (
  "%ProgramFiles%\RedHat\Podman\podman.exe" %*
  exit /b
)

if exist "%LocalAppData%\Programs\Podman\podman.exe" (
  "%LocalAppData%\Programs\Podman\podman.exe" %*
  exit /b
)

echo container-cli: neither docker nor podman was found. 1>&2
exit /b 127
