@echo off
set CLANG="c:\program files\llvm-17.0.1\bin\clang-cl.exe"
set CommonCompilerFlags=/O2
set CommonCompilerFlags=%CommonCompilerFlags% /arch:AVX2
rem set CommonCompilerFlags=%CommonCompilerFlags% -fsanitize=address
set CommonCompilerFlags=%CommonCompilerFlags% /Zi
set CommonCompilerFlags=%CommonCompilerFlags% /W4
set CommonCompilerFlags=%CommonCompilerFlags% /DVADC_API= /DWIN32 /D_CRT_SECURE_NO_WARNINGS
set CommonCompilerFlags=%CommonCompilerFlags% /DNDEBUG
set CommonCompilerFlags=%CommonCompilerFlags% /DTRACY_ENABLE
set CommonCompilerFlags=%CommonCompilerFlags% /DTRACY_NO_SAMPLING
set CommonLinkerFlags=/incremental:no

rem %CLANG% %CommonCompilerFlags% vadc.c /link %CommonLinkerFlags% kernel32.lib lib\onnxruntime.lib

rem %CLANG% %CommonCompilerFlags% filter_script.c /link %CommonLinkerFlags% kernel32.lib

del test.pdb >nul & %CLANG% %CommonCompilerFlags% /MP /Itracy test.c tracy\TracyClient.cpp /link %CommonLinkerFlags% kernel32.lib Shlwapi.lib

del *.obj *.res >nul
