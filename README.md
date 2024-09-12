# README

## Changelog

\[2024-08-17\] README: Examples, ffmpeg-related

\[2024-08-16\] README: Updated description, build instructions, POI

## Up to date description

### Disclaimer

First of all, the main code of the project is ==windows-only== atm. It doesn't have to be though, just didn't get to it yet.

This is a work in progress repo with a lot of parts in half-finished half-POC state. Very little effort went into organization. Think of this as a github mirror of a personal project, which isn't intended for any public use yet at all, so adjust expectations accordingly. You've been warned =)

### Overview

Applications:
- cembed - helper program to embed weights into a C static array
- vadc - main program
- filter_script - helper to make ffmpeg filter script
- test - runs various tests of the C backend

Dirs:
- include - onnxruntime headers
- lib - onnxruntime import library and dlls
- tracy - Tracy profiler
- testdata - test validation files
- tinygrad - was used for old experimental tinygrad implementation of silero v3.1 (silero_tg.py)

POI:
- `silero_vad.py` - pytorch implementation for silero vad models v3.1, v4, v5
- `utils.py` - various helpers for .testtensor files, state dict loaders/mapping helpers for official/unofficial silero pytorch models, audio chunking utils.
- `onnx_helpers.h/c` - code specific for onnxruntime backend
- C backend:
    - `silero.h`
    - `maths.h`
    - `tensor.h`
    - `decoder.c`
    - `conv.c`
    - `lstm.c`
    - `transformer.c`

Old and obsolete:
- `vadc.py` - old python wrapper that runs vadc
- `silero_vad_v3.py` - old pytorch implementation for silero vad models. Obsolete
- `silero_tg.py` - old tinygrad implementation for silero vad v3.1

VADC has 2 backends:
- C
- onnxruntime

Call:
- `build_msvc.bat 0` for C backend (default)
- `build_msvc.bat 1` for onnxruntime backend

Onnxruntime backend is much faster and support more options, like batching and custom sequence lengths. But you need to copy the onnxruntime dlls yourself to where vadc.exe is. Also, batching is not supported by official Silero onnx files, you have to export minibatch-enabled onnx models from pytorch code in `silero_vad.py`, and the export helper code isn't added to git yet.

### Build instructions

Requirements to build the C programs:
- windows
- msvc 2022
- avx2 - there are some crude simd optimizations, can be disabled

How to build:
run `build_msvc.bat`

`build.bat` is old and used hardcoded clang paths, is not updated and probably doesn't work.

The main executable is vadc. It calls ffmpeg with proper parameters if you pass a valid filepath, so have ffmpeg in PATH.
Default backend is C. At the moment it is implementation of Silero VAD v3.1 16kHz.

When built with C backend (the default) the vadc executable should be self-sufficient (not counting ffmpeg) and has the weights embedded.

Usage:
`vadc.exe <filepath>`

Should output speech segment timestamps to stdout.

**Note:** only timestamps/probabilities are printed to stdout, so you can redirect them to files or other programs. All errors, warnings, statistics and diagnostics are printed to stderr.

There are tests which you can run with test.exe, but 6 of them should fail with max error magnitude 0 because their validation data is not added to the git repo because of the size or length of the test.

### ffmpeg support
If filepath is passed to vadc, it will attempt to call ffmpeg to automatically convert audio from the provided media filepath to a suitable format. Doesn't check if ffmpeg is available yet, so put it in PATH or near `vadc.exe`.

If no filepath passed, vadc will read audio data from stdin instead, assuming it is already in a suitable format (raw samples, 16kHz rate, mono, PCM s16le).

By default, the first audio stream in the media is processed. Use the `--audio_source` argument to change that (0-based audio source index). Formatted to ffmpeg argument like so: `-map 0:a:%d`

Specify `--start_seconds` to seek in the media file. Formatted to ffmpeg argument like so: `-ss %f`

### Examples

#### short and quick
`vadc podcast.mp3`
calls ffmpeg to decode `podcast.mp3` internally

#### multichannel audio
`vadc multichannel_video.mp4 --audio_source 1`
process 2nd audio channel

#### seeking
`vadc --start_seconds 60.5 podcast.mp3`
skip first 60.5 seconds of the audio.

#### preprocessed audio
`ffmpeg -y -hide_banner -loglevel error -i input.mp3 -vn -af asetpts=N/SR/TB -c:a pcm_s16le -ac 1 -ar 16000 -sample_fmt s16 -f s16le raw_audio.s16le`
`vadc < raw_audio.s16le`
write preprocessed audio to `raw_audio.s16le` then pipe it to vadc stdin. useful for profiling, as it eliminates the decoding overhead.

#### preprocessed audio piped
`ffmpeg -y -hide_banner -loglevel error -i input.mp3 -vn -af asetpts=N/SR/TB -c:a pcm_s16le -ac 1 -ar 16000 -sample_fmt s16 -f s16le - | vadc`
same as above but in one line and without intermediate files

## TODO: UPDATE OLD README

- [ ] Fix outdated readme contents in the sections below

==WARNING: the rest of the readme below is somewhat outdated==

---
## What it does

Given an input audio, outputs timestamp ranges for segments which are likely to contain speech.

## Quick start

The program expects raw PCM 16-bit signed integer audio in 16kHz, 1 channel, to be piped into stdin. It will output the timestamps of speech segments to stdout.

You can use ffmpeg to convert your audio to the correct format expected by the program from any audio or video format supported by ffmpeg. This command will output speech timestamps to a file from the input file "input.mp3":

`ffmpeg -y -hide_banner -loglevel error -i input.mp3 -vn -af asetpts=N/SR/TB -c:a pcm_s16le -ac 1 -ar 16000 -sample_fmt s16 -f s16le - | vadc > timestamps.txt`

## Filterscript

There is a helper program filter_script which can be used to generate a filterscript for ffmpeg which can be used to remove all non-speech audio from an input file.

`type timestamps.txt | filter_script > filter_script.txt`

Optionally, you can append dynaudnorm filter to normalize volume of the resulting audio
`echo , dynaudnorm=f=75:g=21 >> filter_script.txt`

And this is ffmpeg command to use the filterscript to produce the final trimmed down audio:
`ffmpeg -y -hide_banner -loglevel error -stats -i input.mp3 -vn -filter_script:a filter_script.txt -acodec libopus -b:a 48k output.opus`

## Command line options

`--threshold`: Speech probability threshold. Audio segments with probability above this value are considered to contain speech. Higher values increase false negatives, lower values increase false positives. Default: 0.5.
`--neg_threshold_relative`: Threshold for silence relative to speech threshold specified by `--threshold`. Higher values reduce false positives. Default: 0.15.

The probability of what is considered silence or speech depends on whether we are in a speech segment or not. If we are in a non-speech segment, then the probability of speech to start a speech segment is `--threshold`. If we are in a speech segment, then the probability of speech to start a speech segment is `--threshold` - `--neg_threshold_relative`.

`--min_silence`: Minimum silence duration in milliseconds to consider the end of a speech segment. Default: 200ms.
If we have at least 200ms of silence, end the current speech segment.
`--min_speech`: Minimum speech duration in milliseconds to consider the start of a speech segment. Default: 250ms.
If we have at least 250ms of speech, start a new speech segment.

Threshold and negative threshold relative are used to define the range of probabilities considered to be speech. For example, if `--threshold` is set to 0.5 and `--neg_threshold_relative` is set to 0.15, then the range is 0.35 to 0.5.

Putting this all together using the default values, we look for first 250ms of audio chunks with a probability above 0.5 to start a speech segment. After we kicked into speech, we look for 200ms of audio chunks with probability lower than 0.35 to end the speech segment.

`--speech_pad`: Padding in milliseconds added to start and end of each speech segment. Reduces clipping. Default: 30ms.
This is done after the speech segments have been detected. The speech segments are padded by 30ms on each side, extending them by shrinking the silence segments between them.

*Note: since default padding is 30ms and default `--min_silence` is 200ms, when using the default values there is no risk of padding collisions. There is code that checks for this case, but it was never explicitly tested.*

`--raw_probabilities`: Output raw speech/silence classification probabilities for each audio chunk. This is mostly useful for debugging.

`--model`: Specify explicit path to model. Ignored in C backend. Supports silero v3, v4 and v5 (`silero_vad_v3.onnx` and `silero_vad_v4.onnx`).

`--stats`: prints speed and detected speech durations to stderr

`--sequence_count`: if the model supports variable sequence count (v3, v4), can specify it here. Ignored in C backend.

16kHz (multiples of 256):
- 512, 768, 1024, 1280, 1536
(not supported yet) 8kHz (multiples of 128):
- 256, 384, 512, 640, 768

`--batch`: if the model supports it, can specify batch/minibatch count with this option. Ignored in C backend.

`--output_centi_seconds`: output integer timestamps, in 1/100ths of second. In other words, divide by 100 to get seconds.
with this option:
- `691,774`
- `1027,1120`
without this option (default):
- `6.91,7.74`
- `10.27,11.20`

## More info

Silero model outputs probability for each chunk of audio. One chunk is 1536 samples long, which at 16kHz is 0.096 seconds, or 96ms long.
This also means that `--min_silence` and `--min_speech` are rounded to the closest multiple of 96, with a minimum of 1. This is literally the code that does this:

```cpp
// float min_silence_duration_ms, chunk_duration_ms;
int min_silence_duration_chunks = min_silence_duration_ms / chunk_duration_ms + 0.5f;
if (min_silence_duration_chunks < 1)
{
    min_silence_duration_chunks = 1;
}
```

Which means default values of 200ms and 250ms are in practice 192ms and 288ms.
*The `--speech_pad` is on the other hand used as is, since it is applied after the chunks have been processed.*

Also most of the command line options were not tested with absurdly large, small, negative or even 0 values.
