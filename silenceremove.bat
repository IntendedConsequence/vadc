@REM @echo off
setlocal

set sourcefilepath=%~dpnx1

set timestamps=%~n1.timestamps.txt
set filter_script=%~n1.filter_script.txt

ffmpeg -y -hide_banner -loglevel error -i "%sourcefilepath%" -vn -af asetpts=N/SR/TB -c:a pcm_s16le -ac 1 -ar 16000 -sample_fmt s16 -f s16le - | vadc.exe > "%timestamps%"

type "%timestamps%" | filter_script.exe > "%filter_script%"
echo , dynaudnorm=f=75:g=21 >> "%filter_script%"

ffmpeg -y -hide_banner -loglevel error -stats -i "%sourcefilepath%" -vn -filter_script:a "%filter_script%" -acodec libopus -b:a 48k "%~n1_silero.opus"

endlocal
