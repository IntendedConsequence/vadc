@echo off
setlocal enabledelayedexpansion

where /Q cl.exe || (
  set __VSCMD_ARG_NO_LOGO=1
  for /f "tokens=*" %%i in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath') do set VS=%%i
  if "!VS!" equ "" (
    echo ERROR: Visual Studio installation not found
    exit /b 1
  )
  call "!VS!\VC\Auxiliary\Build\vcvarsall.bat" amd64 || exit /b 1
)

if "%VSCMD_ARG_TGT_ARCH%" neq "x64" (
  echo ERROR: please run this from MSVC x64 native tools command prompt, 32-bit target is not supported!
  exit /b 1
)

set CL=/W4 /WX /Zi /O1 /diagnostics:caret /options:strict /D_CRT_SECURE_NO_WARNINGS
set LINK=/INCREMENTAL:NO /SUBSYSTEM:CONSOLE kernel32.lib

cl.exe /nologo vadc.c /link lib\onnxruntime.lib
cl.exe /nologo filter_script.c /link
del *.obj *.res >nul
