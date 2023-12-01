# README

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

The probability of what is considered silence or speech depends on wether we are in a speech segment or not. If we are in a non-speech segment, then the probability of speech to start a speech segment is `--threshold`. If we are in a speech segment, then the probability of speech to start a speech segment is `--threshold` - `--neg_threshold_relative`.

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
