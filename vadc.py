import io
import subprocess
from pathlib import Path
import sys
import click

FFMPEG_PATH = "ffmpeg"
VADC_PATH = "vadc"


# TODO(irwin):
# - seek
# - pipe in media
def run_vadc(input_file: Path,
                 threshold: float=0.5,
                 neg_threshold_relative: float=0.15,
                 min_silence: int=200,
                 min_speech: int=250,
                 speech_pad: int=30,
                 raw_probabilities: bool=False,
                 audio_stream: int=0,
                 debug_no_buffer: bool=False):
    # Prepare ffmpeg command
    format_options = f"-vn -sn -dn -map 0:a:{audio_stream} -af asetpts=N/SR/TB -c:a pcm_s16le -ac 1 -ar 16000 -sample_fmt s16 -f s16le -"
    input_file_argument = f"\"{input_file}\"" if input_file != '-' else '-'
    ffmpeg_cmd = f"\"{FFMPEG_PATH}\" -y -hide_banner -loglevel error -i {input_file_argument} {format_options}"
    # print(ffmpeg_cmd)
    # print(shlex.split(ffmpeg_cmd))

    # Prepare vadc command
    vadc_cmd = f"\"{VADC_PATH}\" --threshold {threshold} --neg_threshold_relative {neg_threshold_relative} --min_silence {min_silence} --min_speech {min_speech} --speech_pad {speech_pad}"
    if raw_probabilities:
        vadc_cmd += " --raw_probabilities"

    if input_file == '-':
        # obtain this process' stdin to pass to subprocess.Popen
        current_stdin_handle = sys.stdin.fileno()

    # Start ffmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stdin=current_stdin_handle if input_file == '-' else None)

    # Start vadc process
    vadc_process = subprocess.Popen(vadc_cmd, stdin=ffmpeg_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if raw_probabilities and not debug_no_buffer:
        # Use an io.BufferedReader for efficient reading of large chunks of data
        buffer = io.BufferedReader(vadc_process.stdout)
        partial_line = ''

        while True:
            chunk = buffer.read(256)  # Read in 4KB chunks
            if not chunk:
                break

            chunk = partial_line + chunk.decode('utf-8')
            lines = chunk.splitlines(keepends=True)

            try:
                partial_line = lines.pop()  # Save the last line in case it's incomplete
            except IndexError:
                pass

            # print(len(lines), file=sys.stderr)
            for line in lines:  # Process all complete non-empty lines
                stripped = line.strip()
                if stripped:
                    yield stripped

        ffmpeg_process.wait()
        vadc_process.wait()

        # Ensure any remaining data in partial_line is processed
        if partial_line.strip():
            yield partial_line.strip()
    else:
        # Yield output from vadc in real-time
        while True:
            line = vadc_process.stdout.readline()
            if not line:
                break
            yield line.decode('utf-8').strip()

        ffmpeg_process.wait()
        vadc_process.wait()

@click.command(context_settings=dict(show_default=True))
@click.argument('input_file', type=click.Path(exists=True, path_type=Path, allow_dash=True))
@click.option('--threshold', type=float, default=0.5)
@click.option('--neg_threshold_relative', type=float, default=0.15)
@click.option('--min_silence', type=int, default=200)
@click.option('--min_speech', type=int, default=250)
@click.option('--speech_pad', type=int, default=30)
@click.option('--raw_probabilities', is_flag=True)
def main(input_file: Path,
         threshold: float,
         neg_threshold_relative: float,
         min_silence: int,
         min_speech: int,
         speech_pad: int,
         raw_probabilities: bool):

    options = {
        "input_file": input_file,
        "threshold": threshold,
        "neg_threshold_relative": neg_threshold_relative,
        "min_silence": min_silence,
        "min_speech": min_speech,
        "speech_pad": speech_pad,
        "raw_probabilities": raw_probabilities
    }

    if False:
        buffered = [o for o in run_vadc(**options, debug_no_buffer=False)]
        line_by_line = [o for o in run_vadc(**options, debug_no_buffer=True)]

        # diff the two lists using python's difflib
        import difflib
        diff = difflib.unified_diff(buffered, line_by_line)
        for line in diff:
            print(line, file=sys.stderr)
    else:
        for o in run_vadc(**options, debug_no_buffer=False):
            print(o, file=sys.stderr)

if __name__ == "__main__":
    main()

