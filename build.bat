@echo off
set CLANG="c:\program files\llvm\bin\clang-cl.exe"
set CommonCompilerFlags=/Od
rem set CommonCompilerFlags=%CommonCompilerFlags% -fsanitize=address
set CommonCompilerFlags=%CommonCompilerFlags% /Zi
set CommonCompilerFlags=%CommonCompilerFlags% /W4
set CommonLinkerFlags=/incremental:no

%CLANG% %CommonCompilerFlags% vadc.c /link %CommonLinkerFlags% kernel32.lib lib\onnxruntime.lib

%CLANG% %CommonCompilerFlags% filter_script.cpp /link %CommonLinkerFlags% kernel32.lib
