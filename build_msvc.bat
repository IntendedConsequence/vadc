@echo off
setlocal enabledelayedexpansion

set BUILD_CACHE=%~dp0\_build_cache.cmd

if exist "!BUILD_CACHE!" (
  rem cache file exists, so call it to set env variables very fast
  call "!BUILD_CACHE!"
) else (
  set __VSCMD_ARG_NO_LOGO=1
  for /f "tokens=*" %%i in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath') do set VS=%%i
  if "!VS!" equ "" (
    echo ERROR: Visual Studio installation not found
    exit /b 1
  )
  call "!VS!\VC\Auxiliary\Build\vcvarsall.bat" amd64 || exit /b 1

  echo set PATH=!PATH!> "!BUILD_CACHE!"
  echo set INCLUDE=!INCLUDE!>> "!BUILD_CACHE!"
  echo set LIB=!LIB!>> "!BUILD_CACHE!"
  echo set VSCMD_ARG_TGT_ARCH=!VSCMD_ARG_TGT_ARCH!>> "!BUILD_CACHE!"

  rem Depending on whether you are build .NET or other stuff, there are more
  rem env variables you might want to add to cache, like:
  rem Platform, FrameworkDir, NETFXSDKDir, WindowsSdkDir, WindowsSDKVersion, VCINSTALLDIR, ...
)

rem put your build commands here

if "%VSCMD_ARG_TGT_ARCH%" neq "x64" (
  echo ERROR: please run this from MSVC x64 native tools command prompt, 32-bit target is not supported!
  exit /b 1
)

set CL=/W4 /WX /Zi /O1 /diagnostics:caret /options:strict /DWIN32 /D_CRT_SECURE_NO_WARNINGS
@rem set CL=%CL% /fsanitize=address
set LINK=/INCREMENTAL:NO /SUBSYSTEM:CONSOLE kernel32.lib Shlwapi.lib

cl.exe /nologo vadc.c /link lib\onnxruntime.lib
cl.exe /nologo filter_script.c /link
cl.exe /nologo test.c /link
cl.exe /nologo /MD decoder.c /link /DLL /OUT:decoder.dll
del *.obj *.res >nul
