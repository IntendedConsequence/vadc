#include "vadc.h"

#include <inttypes.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h> // GetModuleFileNameW
#include <Shlwapi.h> // PathRemoveFileSpecW, PathAppendW

#include "string8.c"

#include <tracy\TracyC.h>

#include "utils.h"

#if ONNX_INFERENCE_ENABLED
#include "onnx_helpers.c"
#else
#include "silero.h"
#endif // ONNX_INFERENCE_ENABLED


#define MEMORY_IMPLEMENTATION
#include "memory.h"

#ifndef DEBUG_WRITE_STATE_TO_FILE
#define DEBUG_WRITE_STATE_TO_FILE 0
#endif


// TODO(irwin):
// - move win32-specific stuff to separate file


#if DEBUG_WRITE_STATE_TO_FILE
typedef struct DEBUG_Silero_State DEBUG_Silero_State;
struct DEBUG_Silero_State
{
   float samples[1536];
   float state_h[128];
   float state_c[128];
};

static FILE *getDebugFile()
{
   static FILE *debug_file = NULL;
   if ( debug_file == NULL )
   {
      debug_file = fopen( "debug_state.out", "wb" );
   }
   return debug_file;
}
#endif



void process_chunks( MemoryArena *arena, VADC_Context context, Silero_Config config,
                    const size_t buffered_samples_count,
                    const float *samples_buffer_float32,
                    float *probabilities_buffer)
{

   VAR_UNUSED(arena);
   {
      int stride = (int)context.buffers.window_size_samples * config.batch_size;
      for (size_t offset = 0;
         offset < buffered_samples_count;
         offset += stride)
      {
         // NOTE(irwin): copy a slice of the buffered samples
         size_t samples_count_left = buffered_samples_count - offset;
         size_t samples_count = samples_count_left > stride ? stride : samples_count_left;

         // TODO(irwin): memset to 0 the entire tensor for simplicity, to avoid manual error-prone padding offset calculations
         memset( context.buffers.input_samples, 0, stride * sizeof( context.buffers.input_samples[0] ) );
         memmove( context.buffers.input_samples, samples_buffer_float32 + offset, samples_count * sizeof( context.buffers.input_samples[0] ) );

         // NOTE(irwin): pad chunks with not enough samples
         // for ( size_t pad_index = samples_count; pad_index < stride; ++pad_index )
         // {
         //    context.buffers.input_samples[pad_index] = 0.0f;
         // }

         memmove( context.buffers.lstm_h, context.buffers.lstm_h_out, context.buffers.lstm_count * sizeof( context.buffers.lstm_h[0] ) );
         memmove( context.buffers.lstm_c, context.buffers.lstm_c_out, context.buffers.lstm_count * sizeof( context.buffers.lstm_c[0] ) );

         int output_stride = config.output_stride;

         // NOTE(irwin): if there aren't enough samples for a full batch it can only be the very last batch
         //              otherwise we don't tell the backend that the batch size is lower, and so if one
         //              batch in the middle of the stream comes shorter, the lstm state will get screwed up
         //              from processing null samples in the middle of the stream.
         backend_run(arena, &context, config);

         // NOTE(irwin): will copy stale probabilities from previous batch if not enough samples for a full batch
         for (int i = 0; i < config.batch_size; ++i)
         {
            float result_probability = context.buffers.output[i * output_stride + config.silero_probability_out_index];
            *probabilities_buffer++ = result_probability;
         }

      }
   }
}

void process_chunks_v5( MemoryArena *arena, VADC_Context context, Silero_Config config,
                    const size_t buffered_samples_count,
                    const float *samples_buffer_float32,
                    float *probabilities_buffer)
{

   VAR_UNUSED(arena);
   {
      s32 context_size = config.context_size;
      s32 window_size = config.input_count;
      s32 total_sequence_count = context_size + window_size;

      int input_buffer_size = ((int)context.buffers.window_size_samples + context_size) * config.batch_size;
      int stride = (int)context.buffers.window_size_samples * config.batch_size;
      for (size_t offset = 0;
         offset < buffered_samples_count;
         offset += stride)
      {
         // NOTE(irwin): copy context from previous batch, should be zeroes for first batch due to ZII
         memmove(context.buffers.input_samples, context.buffers.input_samples + input_buffer_size - context_size, context_size * sizeof(float));

         // NOTE(irwin): copy a slice of the buffered samples
         // size_t samples_count_left = buffered_samples_count - offset;
         // size_t samples_count = samples_count_left > stride ? stride : samples_count_left;

         memmove(context.buffers.input_samples + context_size, samples_buffer_float32 + offset, window_size * sizeof(float));
         for (int batch_index = 1; batch_index < config.batch_size; ++batch_index)
         {
            memmove(context.buffers.input_samples + (batch_index * total_sequence_count),
                    (samples_buffer_float32 + offset) + (batch_index * window_size) - context_size,
                    64 * sizeof(float));

            memmove(context.buffers.input_samples + (batch_index * total_sequence_count) + context_size,
                    (samples_buffer_float32 + offset) + (batch_index * window_size),
                    512 * sizeof(float));
         }

         memmove( context.buffers.lstm_h, context.buffers.lstm_h_out, context.buffers.lstm_count * sizeof( context.buffers.lstm_h[0] ) );
         memmove( context.buffers.lstm_c, context.buffers.lstm_c_out, context.buffers.lstm_count * sizeof( context.buffers.lstm_c[0] ) );

         int output_stride = config.output_stride;

         // NOTE(irwin): if there aren't enough samples for a full batch it can only be the very last batch
         //              otherwise we don't tell the backend that the batch size is lower, and so if one
         //              batch in the middle of the stream comes shorter, the lstm state will get screwed up
         //              from processing null samples in the middle of the stream.
         backend_run(arena, &context, config);

         // NOTE(irwin): will copy stale probabilities from previous batch if not enough samples for a full batch
         for (int i = 0; i < config.batch_size; ++i)
         {
            float result_probability = context.buffers.output[i * output_stride + config.silero_probability_out_index];
            *probabilities_buffer++ = result_probability;
         }

      }
   }
}


FeedProbabilityResult feed_probability(FeedState *state,
                      int min_silence_duration_chunks,
                      int min_speech_duration_chunks,
                      float probability,
                      float threshold,
                      float neg_threshold,
                      int global_chunk_index
                      )
{
   FeedProbabilityResult result = {0};

   if (probability >= threshold && state->temp_end > 0)
   {
      state->temp_end = 0;
   }

   if (!state->triggered)
   {

      if (probability >= threshold)
      {
         state->triggered = 1;
         state->current_speech_start = global_chunk_index;
      }
   }
   else
   {
      if (probability < neg_threshold)
      {
         if (state->temp_end == 0)
         {
            state->temp_end = global_chunk_index;
         }
         if (global_chunk_index - state->temp_end < min_silence_duration_chunks)
         {

         }
         else
         {

            if (state->temp_end - state->current_speech_start >= min_speech_duration_chunks)
            {
               result.speech_start = state->current_speech_start;
               result.speech_end = state->temp_end;
               result.is_valid = 1;
            }

            state->current_speech_start = 0;
            state->temp_end = 0;
            state->triggered = 0;
         }
      }
   }


   return result;
}

void emit_speech_segment(FeedProbabilityResult segment,
                         float speech_pad_ms,
                         Segment_Output_Format output_format,
                         VADC_Stats *stats,
                         float seconds_per_chunk)
{
   const float spc = seconds_per_chunk;

   const float speech_pad_s = speech_pad_ms / 1000.0f;

   float speech_end_padded = (segment.speech_end * spc) + speech_pad_s;

   // NOTE(irwin): print previous start/end times padded in seconds
   float speech_start_padded = (segment.speech_start * spc) - speech_pad_s;
   if (speech_start_padded < 0.0f)
   {
      speech_start_padded = 0.0f;
   }

   stats->total_speech += (double)speech_end_padded - (double)speech_start_padded;

   switch (output_format)
   {
      case Segment_Output_Format_Seconds:
      {
         fprintf(stdout, "%.2f,%.2f\n", speech_start_padded, speech_end_padded);
      } break;

      case Segment_Output_Format_CentiSeconds:
      {
         s64 start_centi = (s64)((double)speech_start_padded * 100.0 + 0.5);
         s64 end_centi = (s64)((double)speech_end_padded * 100.0 + 0.5);
         fprintf(stdout, "%" PRId64 "," "%" PRId64 "\n", start_centi, end_centi);
      } break;
   }
   fflush(stdout);
   print_speech_stats(*stats);
}

FeedProbabilityResult combine_or_emit_speech_segment(FeedProbabilityResult buffered, FeedProbabilityResult feed_result,
                                                     float speech_pad_ms, Segment_Output_Format output_format, VADC_Stats *stats,
                                                     float seconds_per_chunk)
{
   FeedProbabilityResult result = buffered;

   const float spc = seconds_per_chunk;


   const float speech_pad_s = speech_pad_ms / 1000.0f;

   float current_speech_start_padded = (feed_result.speech_start * spc) - speech_pad_s;
   if (current_speech_start_padded < 0.0f)
   {
      current_speech_start_padded = 0.0f;
   }

   if (result.is_valid)
   {
      float buffered_speech_end_padded = (result.speech_end * spc) + speech_pad_s;
      if (buffered_speech_end_padded >= current_speech_start_padded)
      {
         result.speech_end = feed_result.speech_end;
      }
      else
      {
         emit_speech_segment(result, speech_pad_ms, output_format, stats, spc);

         result = feed_result;
      }
   }
   else
   {
      result = feed_result;
   }

   return result;
}


#if 0
void read_wav_ffmpeg( const char *fname_inp )
{
   const wchar_t ffmpeg_to_s16le[] = L"ffmpeg -hide_banner -loglevel error -stats -i \"%s\" -map 0:a:0 -vn -sn -dn -ac 1 -ar 16k -f s16le -";
   wchar_t *fname_widechar = nullptr;
   if ( UTF8_ToWidechar( &fname_widechar, fname_inp, 0 ) )
   {
      wchar_t ffmpeg_final[4096];
      swprintf( ffmpeg_final, 4096, ffmpeg_to_s16le, fname_widechar );

      free( fname_widechar );

      // Create the pipe
      SECURITY_ATTRIBUTES saAttr = {sizeof( SECURITY_ATTRIBUTES )};
      saAttr.bInheritHandle = FALSE;

      HANDLE ffmpeg_stdout_read, ffmpeg_stdout_write;

      if ( !CreatePipe( &ffmpeg_stdout_read, &ffmpeg_stdout_write, &saAttr, 0 ) )
      {
         fprintf( stderr, "Error creating ffmpeg pipe\n" );
         return false;
      }

      // NOTE(irwin): ffmpeg does inherit the write handle to its output
      SetHandleInformation( ffmpeg_stdout_write, HANDLE_FLAG_INHERIT, 1 );

      // Launch ffmpeg and redirect its output to the pipe
      STARTUPINFOW startup_info_ffmpeg = {sizeof( STARTUPINFO )};
      // NOTE(irwin): hStdInput is 0, we don't want ffmpeg to inherit our stdin
      startup_info_ffmpeg.hStdOutput = ffmpeg_stdout_write;
      startup_info_ffmpeg.hStdError = GetStdHandle( STD_ERROR_HANDLE );
      startup_info_ffmpeg.dwFlags |= STARTF_USESTDHANDLES;

      PROCESS_INFORMATION ffmpeg_process_info = {};

      if ( !CreateProcessW( NULL, ffmpeg_final, NULL, NULL, TRUE, 0, NULL, NULL, &startup_info_ffmpeg, &ffmpeg_process_info ) )
      {
         fprintf( stderr, "Error launching ffmpeg\n" );
         return false;
      }

      // Close the write end of the pipe, as we're not writing to it
      CloseHandle( ffmpeg_stdout_write );

      // NOTE(irwin): restore non-inheritable status
      SetHandleInformation( ffmpeg_stdout_write, HANDLE_FLAG_INHERIT, 0 );

      // we can close the handles early if we're not going to use them
      CloseHandle( ffmpeg_process_info.hProcess );
      CloseHandle( ffmpeg_process_info.hThread );


      if ( ffmpeg_stdout_read != INVALID_HANDLE_VALUE )
      {
         const int BUFSIZE = 4096 * 2 * 2;
         // Read ffmpeg's output
         unsigned char buffer[BUFSIZE];
         int leftover = 0;

         DWORD dwRead = 0;
         unsigned char *buffer_dst = buffer + leftover;
         auto byte_count_to_read = sizeof( buffer ) - leftover;
         while ( ReadFile( ffmpeg_stdout_read, buffer_dst, byte_count_to_read, &dwRead, NULL ) )
         {
            if ( dwRead == 0 )
            {
               // fflush(stdout);
               // NOTE(irwin): we ignore any leftover bytes in buffer in this case
               break;
            }

            DWORD bytes_in_buffer = dwRead + leftover;
            DWORD remainder = bytes_in_buffer % sizeof( int16_t );

            int16_t *from = (int16_t *)buffer;
            int16_t *to = (int16_t *)(buffer + (bytes_in_buffer - remainder));

            //-----------------------------------------------------------------------------
            // got bytes, do something with them here
            //-----------------------------------------------------------------------------

            if ( remainder != 0 )
            {
               memmove( buffer, to, remainder );
            }
            leftover = remainder;
            //printf( "%.*s", (int)dwRead, buffer );
            // printf("\n%d\n", (int)dwRead);
            // fflush(stdout);
            // WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buffer, dwRead, NULL, NULL);
            // FlushFileBuffers(GetStdHandle(STD_OUTPUT_HANDLE));
            // FlushFileBuffers(GetStdHandle(STD_ERROR_HANDLE));
         }
      }
   }
}
#endif

typedef enum BS_Error BS_Error;
enum BS_Error
{
   BS_Error_NoError = 0,
   BS_Error_Error,
   BS_Error_EndOfFile,
   BS_Error_Memory,
   BS_Error_CantOpenFile,

   BS_Error_COUNT
};


typedef struct Buffered_Stream Buffered_Stream;
typedef BS_Error ( *Refill_Function ) (Buffered_Stream *);


struct Buffered_Stream
{
   u8 *start;
   u8 *cursor;
   u8 *end;


   Refill_Function refill;

   BS_Error error_code;

   // NOTE(irwin): win32
   HANDLE read_handle_internal;

   // NOTE(irwin): crt
   FILE *file_handle_internal;

   u8 *buffer_internal;
   size_t buffer_internal_size;
};

BS_Error refill_zeros(Buffered_Stream *s)
{
   static u8 zeros[256] = {0};

   s->start = zeros;
   s->cursor = zeros;
   s->end = zeros + sizeof( zeros );

   return s->error_code;
}

static BS_Error fail_buffered_stream(Buffered_Stream *s, BS_Error error_code)
{
   s->error_code = error_code;
   s->refill = refill_zeros;
   s->refill( s );

   return s->error_code;
}

BS_Error refill_FILE( Buffered_Stream *s )
{
   if (s->cursor == s->end)
   {
      size_t values_read = fread( s->buffer_internal, 1, s->buffer_internal_size, s->file_handle_internal );
      if (values_read == s->buffer_internal_size)
      {
         s->start = s->buffer_internal;
         s->cursor = s->buffer_internal;
         s->end = s->start + values_read;
      }
      else if ( values_read > 0 )
      {
         s->start = s->buffer_internal;
         s->cursor = s->buffer_internal;
         s->end = s->start + values_read;
      }
      else
      {
         if (feof(s->file_handle_internal))
         {
            return fail_buffered_stream( s, BS_Error_EndOfFile );
         }
         else if (ferror(s->file_handle_internal))
         {
            return fail_buffered_stream( s, BS_Error_Error );
         }
      }
   }

   return s->error_code;
}

BS_Error refill_HANDLE( Buffered_Stream *s )
{
   if ( s->cursor == s->end )
   {
      DWORD byte_count_read = 0;
      DWORD byte_count_read_total = 0;
      BOOL read_file_result = 0;
      do
      {
         // NOTE(irwin): keep calling ReadFile until we've filled our internal buffer or until ReadFile returns 0
         u8 *destination = s->buffer_internal + byte_count_read_total;
         DWORD max_byte_count_to_read = (DWORD)s->buffer_internal_size - byte_count_read_total;

         read_file_result = ReadFile( s->read_handle_internal, destination, max_byte_count_to_read, &byte_count_read, NULL );
         byte_count_read_total += byte_count_read;
      } while (read_file_result && byte_count_read > 0 && byte_count_read_total < (DWORD)s->buffer_internal_size);

      if ( byte_count_read_total > 0 )
      {
         s->start = s->buffer_internal;
         s->cursor = s->buffer_internal;
         s->end = s->start + byte_count_read_total;
      }
      else
      {
         if ( !read_file_result )
         {
            return fail_buffered_stream( s, BS_Error_EndOfFile );
         }
         else // read_file_result && bytes_read == 0
         {
            return fail_buffered_stream( s, BS_Error_Error );
         }
      }
   }

   return s->error_code;
}

static void init_buffered_stream_ffmpeg(MemoryArena *arena, Buffered_Stream *s, String8 fname_inp, size_t buffer_size,
                  int audio_source,
                  float start_seconds)
{
   memset( s, 0, sizeof( *s ) );

   const char *ffmpeg_to_s16le = "ffmpeg -hide_banner -loglevel error -nostats -ss %f -i \"%.*s\" -map 0:a:%d -vn -sn -dn -ac 1 -ar 16k -f s16le -";
   String8 ffmpeg_command = String8_pushf(arena, ffmpeg_to_s16le, start_seconds, fname_inp.size, fname_inp.begin, audio_source);
   wchar_t *ffmpeg_command_wide = NULL;
   String8_ToWidechar(arena, &ffmpeg_command_wide, ffmpeg_command);

   {
      // Create the pipe
      SECURITY_ATTRIBUTES saAttr = {sizeof( SECURITY_ATTRIBUTES )};
      saAttr.bInheritHandle = FALSE;

      HANDLE ffmpeg_stdout_read, ffmpeg_stdout_write;

      if ( !CreatePipe( &ffmpeg_stdout_read, &ffmpeg_stdout_write, &saAttr, 0 ) )
      {
         fprintf( stderr, "Error creating ffmpeg pipe\n" );
         return;
      }

      // NOTE(irwin): ffmpeg does inherit the write handle to its output
      SetHandleInformation( ffmpeg_stdout_write, HANDLE_FLAG_INHERIT, 1 );

      // Launch ffmpeg and redirect its output to the pipe
      STARTUPINFOW startup_info_ffmpeg = {sizeof( STARTUPINFO )};
      // NOTE(irwin): hStdInput is 0, we don't want ffmpeg to inherit our stdin
      startup_info_ffmpeg.hStdOutput = ffmpeg_stdout_write;
      startup_info_ffmpeg.hStdError = GetStdHandle( STD_ERROR_HANDLE );
      startup_info_ffmpeg.dwFlags |= STARTF_USESTDHANDLES;

      PROCESS_INFORMATION ffmpeg_process_info = {0};

      if ( !CreateProcessW( NULL, ffmpeg_command_wide, NULL, NULL, TRUE, 0, NULL, NULL, &startup_info_ffmpeg, &ffmpeg_process_info ) )
      {
         fprintf( stderr, "Error launching ffmpeg\n" );
         return;
      }

      // Close the write end of the pipe, as we're not writing to it
      CloseHandle( ffmpeg_stdout_write );

      // NOTE(irwin): restore non-inheritable status
      SetHandleInformation( ffmpeg_stdout_write, HANDLE_FLAG_INHERIT, 0 );

      // we can close the handles early if we're not going to use them
      CloseHandle( ffmpeg_process_info.hProcess );
      CloseHandle( ffmpeg_process_info.hThread );


      if ( ffmpeg_stdout_read != INVALID_HANDLE_VALUE )
      {
         // s->buffer_internal = malloc( buffer_size );
         s->buffer_internal = pushSizeZeroed( arena, buffer_size, TEMP_DEFAULT_ALIGNMENT );
         if ( s->buffer_internal )
         {
            // memset( s->buffer_internal, 0, buffer_size );
            s->read_handle_internal = ffmpeg_stdout_read;
            s->refill = refill_HANDLE;
            s->buffer_internal_size = buffer_size;
            s->error_code = BS_Error_NoError;
            s->refill( s );
         }
         else
         {
            fail_buffered_stream( s, BS_Error_Memory );
         }
      }
      else
      {
         // TODO(irwin):
         fail_buffered_stream( s, BS_Error_Error );
      }
   }
}

static void init_buffered_stream_stdin(MemoryArena *arena, Buffered_Stream *s, size_t buffer_size)
{
   memset( s, 0, sizeof( *s ) );
   s->buffer_internal = pushSizeZeroed( arena, buffer_size, TEMP_DEFAULT_ALIGNMENT );
   if ( s->buffer_internal )
   {
      s->read_handle_internal = GetStdHandle(STD_INPUT_HANDLE);
      s->refill = refill_HANDLE;
      s->buffer_internal_size = buffer_size;
      s->error_code = BS_Error_NoError;
      s->refill( s );
   }
   else
   {
      fail_buffered_stream( s, BS_Error_Memory );
   }
}
static void init_buffered_stream_file(MemoryArena *arena, Buffered_Stream *s, FILE *f, size_t buffer_size)
{
   memset( s, 0, sizeof( *s ) );
   if (f)
   {
      // s->buffer_internal = malloc( buffer_size );
      s->buffer_internal = pushSizeZeroed( arena, buffer_size, TEMP_DEFAULT_ALIGNMENT );
      if ( s->buffer_internal )
      {
         // memset( s->buffer_internal, 0, buffer_size );
         s->file_handle_internal = f;
         s->refill = refill_FILE;
         s->buffer_internal_size = buffer_size;
         s->error_code = BS_Error_NoError;
         s->refill( s );
      }
      else
      {
         fail_buffered_stream( s, BS_Error_Memory );
      }
   }
   else
   {
      fail_buffered_stream( s, BS_Error_CantOpenFile );
   }
}

static void deinit_buffered_stream_file( Buffered_Stream *s )
{
   if ( s->file_handle_internal )
   {
      s->file_handle_internal = NULL;
   }

   if ( s->buffer_internal )
   {
      // free( s->buffer_internal );
      s->buffer_internal = NULL;
      s->buffer_internal_size = 0;
   }
}


int run_inference(String8 model_path_arg,
                  MemoryArena *arena,
                  float min_silence_duration_ms,
                  float min_speech_duration_ms,
                  float threshold,
                  float neg_threshold,
                  float speech_pad_ms,
                  float desired_sequence_count,
                  b32 raw_probabilities,
                  Segment_Output_Format output_format,
                  String8 filename,
                  b32 stats_output_enabled,
                  s32 preferred_batch_size,
                  int audio_source,
                  float start_seconds )
{
   Silero_Config config = {0};
   config.batch_size_restriction = 1;
   config.batch_size = 1;

   void *backend = backend_init( arena, model_path_arg, &config );

   if ( !backend )
   {
      return -1;
   }

   b32 is_silero_v5 = config.is_silero_v5;
   if (is_silero_v5)
   {
      fprintf(stderr, "%s", "Model arch is Silero v5\n");
      config.context_size = SILERO_V5_CONTEXT_SIZE;
   }

   if (config.output_dims == 3)
   {
      config.silero_probability_out_index = 1;
      config.output_stride = 2;
   }
   else
   {
      config.silero_probability_out_index = 0;
      config.output_stride = 1;
   }

   config.batch_size = (config.batch_size_restriction == -1) ? preferred_batch_size : config.batch_size_restriction;
   fprintf(stderr, "Running with batch size %d\n", config.batch_size);

   {
      Assert(config.output_dims == 2 || config.output_dims == 3);
      if (config.output_dims == 2)
      {
         config.prob_shape_count = 2;
         config.prob_shape[0] = config.batch_size;
         config.prob_shape[1] = 1;
      }
      else
      {
         config.prob_shape_count = 3;
         config.prob_shape[0] = config.batch_size;
         config.prob_shape[1] = 2;
         config.prob_shape[2] = 1;
      }

      size_t prob_tensor_element_count = 1;
      for (int i = 0; i < config.prob_shape_count; ++i)
      {
         prob_tensor_element_count *= config.prob_shape[i];
      }
      config.prob_tensor_element_count = prob_tensor_element_count;
   }

   {
      int sequence_count = (int)desired_sequence_count;
      if (sequence_count < config.input_size_min)
      {
         sequence_count = config.input_size_min;
      }
      if (sequence_count > config.input_size_max)
      {
         sequence_count = config.input_size_max;
      }
      config.input_count = (s32)sequence_count;
      fprintf(stderr, "Running with sequence count %d\n", config.input_count);
   }

   const float HARDCODED_CHUNK_DURATION_MS = config.input_count / (float)HARDCODED_SAMPLE_RATE * 1000.0f;

   int min_speech_duration_chunks = (int)(min_speech_duration_ms / HARDCODED_CHUNK_DURATION_MS + 0.5f);
   if (min_speech_duration_chunks < 1)
   {
      min_speech_duration_chunks = 1;
   }

   int min_silence_duration_chunks = (int)(min_silence_duration_ms / HARDCODED_CHUNK_DURATION_MS + 0.5f);
   if (min_silence_duration_chunks < 1)
   {
      min_silence_duration_chunks = 1;
   }


   // NOTE(irwin): create tensors and allocate tensors backing memory buffers
   Tensor_Buffers buffers = {0};
   buffers.window_size_samples = (int)config.input_count;

   if (is_silero_v5)
   {
      buffers.input_samples = pushArray(arena, (buffers.window_size_samples + config.context_size) * config.batch_size, float);
   }
   else
   {
      buffers.input_samples = pushArray(arena, buffers.window_size_samples * config.batch_size, float);
   }

   buffers.output = pushArray(arena, config.prob_tensor_element_count, float);

   buffers.lstm_count = 128;
   buffers.lstm_h = pushArray(arena, buffers.lstm_count, float);
   buffers.lstm_c = pushArray(arena, buffers.lstm_count, float);

   buffers.lstm_h_out = pushArray(arena, buffers.lstm_count, float);
   buffers.lstm_c_out = pushArray(arena, buffers.lstm_count, float);

   backend_create_tensors(config, backend, buffers);

   // NOTE(irwin): read samples from a file or stdin and run inference
   // NOTE(irwin): at 16000 sampling rate, one chunk is 96 ms or 1536 samples
   // NOTE(irwin): chunks count being 96, the same as one chunk's length in milliseconds,
   // is purely coincidental
   const int chunks_count = 96;
   // NOTE(irwin): buffered_samples_count is the normalization window size
   const size_t buffered_samples_count = buffers.window_size_samples * chunks_count;

   short *samples_buffer_s16 = pushArray(arena, buffered_samples_count, short);
   float *samples_buffer_float32 = pushArray(arena, buffered_samples_count, float);
   float *probabilities_buffer = pushArray(arena, chunks_count, float);

   Buffered_Stream read_stream = {0};

   size_t buffered_samples_size_in_bytes = sizeof( short ) * buffered_samples_count;
   if (filename.size)
   {
      init_buffered_stream_ffmpeg(arena, &read_stream, filename, buffered_samples_size_in_bytes,
                  audio_source,
                  start_seconds );
   }
   else
   {
      init_buffered_stream_stdin(arena, &read_stream, buffered_samples_size_in_bytes );
   }


   VADC_Context context =
   {
      .backend = backend,
      .buffers = buffers,
   };

   FeedState state = {0};
   int global_chunk_index = 0;

   FeedProbabilityResult buffered = {0};

   VADC_Stats stats = {0};
   stats.output_enabled = stats_output_enabled;
   {
      LARGE_INTEGER frequency = {0};
      LARGE_INTEGER first_timestamp = {0};

      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&first_timestamp);

      stats.first_call_timestamp = first_timestamp.QuadPart;
      stats.timer_frequency = frequency.QuadPart;
   }

   const float HARDCODED_SECONDS_PER_CHUNK = (float)config.input_count / HARDCODED_SAMPLE_RATE;

   s64 total_samples_read = 0;

   // NOTE(irwin): values_read is only accessed inside the for loop
   size_t values_read = 0;
   for(;;)
   {
      BS_Error read_error_code = 0;

      // TODO(irwin): what do we do about errors that arose in refilling the buffered stream
      // but some data was still read? Like EOF, or closed pipe?

      read_error_code = read_stream.refill( &read_stream );

      values_read = (read_stream.end - read_stream.start) / sizeof(short);
      total_samples_read += values_read;
      stats.total_samples = total_samples_read;
      stats.total_duration = (double)total_samples_read / HARDCODED_SAMPLE_RATE;


      // values_read = fread(samples_buffer_s16, sizeof(short), buffered_samples_count, read_source);
      // fprintf(stderr, "%zu\n", values_read);

      //if (values_read > 0)
      if ( read_error_code == BS_Error_NoError )
      {
         memmove( samples_buffer_s16, read_stream.start, read_stream.end - read_stream.start );
         float max_value = 0.0f;
         for (size_t i = 0; i < values_read; ++i)
         {
            float value = samples_buffer_s16[i];
            float abs_value = value > 0.0f ? value : value * -1.0f;
            if (abs_value > max_value)
            {
               max_value = abs_value;
            }
            samples_buffer_float32[i] = value;
         }
         read_stream.cursor = read_stream.end;
#if 0
         if (max_value > 0.0f)
         {
            for (size_t i = 0; i < values_read; ++i)
            {
               samples_buffer_float32[i] /= max_value;
            }
         }
#else
         {
            for (size_t i = 0; i < values_read; ++i)
            {
               samples_buffer_float32[i] /= 32768.0f;
            }
         }
#endif
         size_t leftover = buffered_samples_count - values_read;
         if (leftover > 0)
         {
            for (size_t i = values_read; i < buffered_samples_count; ++i)
            {
               samples_buffer_float32[i] = 0.0f;
            }
         }
      }
      else
      {
         switch (read_stream.error_code)
         {
            case BS_Error_CantOpenFile:
            {
               fprintf( stderr, "Error: BS_Error_CantOpenFile\n" );
            } break;

            case BS_Error_EndOfFile:
            {
               fprintf( stderr, "Error: BS_Error_EndOfFile\n" );
            } break;

            case BS_Error_Error:
            {
               fprintf( stderr, "Error: BS_Error_Error\n" );
            } break;

            case BS_Error_Memory:
            {
               fprintf( stderr, "Error: BS_Error_Memory\n" );
            } break;

            case BS_Error_NoError:
            {
               fprintf( stderr, "Error: BS_Error_NoError\n" );
            } break;

            default:
            {
               fprintf( stderr, "Error: Unreachable switch case\n" );
            } break;
         }

         break;
      }

      if (is_silero_v5)
      {
         process_chunks_v5( arena, context, config,
                        values_read,
                        samples_buffer_float32,
                        probabilities_buffer);
      }
      else
      {
         process_chunks( arena, context, config,
                        values_read,
                        samples_buffer_float32,
                        probabilities_buffer);
      }

      int probabilities_count = (int)(values_read / (float)config.input_count);
      if (!raw_probabilities)
      {
         for (int i = 0; i < probabilities_count; ++i)
         {
            float probability = probabilities_buffer[i];

            FeedProbabilityResult feed_result = feed_probability(&state,
                          min_silence_duration_chunks,
                          min_speech_duration_chunks,
                          probability,
                          threshold,
                          neg_threshold,
                          global_chunk_index
                          );

         if (feed_result.is_valid)
         {
            buffered = combine_or_emit_speech_segment(buffered, feed_result,
                                                      speech_pad_ms, output_format, &stats, HARDCODED_SECONDS_PER_CHUNK);
         }

            // printf("%f\n", probability);
            ++global_chunk_index;
         }
      }
      else
      {
         for (int i = 0; i < probabilities_count; ++i)
         {
            float probability = probabilities_buffer[i];
            printf("%f\n", probability);
            ++global_chunk_index;
         }
      }

   }

   // TODO(irwin):
   deinit_buffered_stream_file( &read_stream );

   if (!raw_probabilities)
   {
      // NOTE(irwin): snap last speech segment to actual audio length
      if (state.triggered)
      {
         int audio_length_samples = (int)((global_chunk_index - 1) * config.input_count);
         if (audio_length_samples - (state.current_speech_start * config.input_count) > (min_speech_duration_chunks * config.input_count))
         {
            FeedProbabilityResult final_segment;
            final_segment.is_valid = 1;
            final_segment.speech_start = state.current_speech_start;
            final_segment.speech_end = (int)(audio_length_samples / config.input_count);

            buffered = combine_or_emit_speech_segment(buffered, final_segment,
                                                         speech_pad_ms, output_format, &stats, HARDCODED_SECONDS_PER_CHUNK);
         }
      }

      if (buffered.is_valid)
      {
         emit_speech_segment(buffered, speech_pad_ms, output_format, &stats, HARDCODED_SECONDS_PER_CHUNK);
      }
   }

   print_speech_stats(stats);

   // g_ort->ReleaseValue(output_tensor);
   // g_ort->ReleaseValue(input_tensor);
   // return ret;
   return 0;
}

static inline void print_speech_stats(VADC_Stats stats)
{
#if 0
   VAR_UNUSED(stats);
#else
   LARGE_INTEGER current;
   QueryPerformanceCounter(&current);
   s64 current_timestamp = current.QuadPart;

   double total_speech = stats.total_speech;
   double total_duration = stats.total_duration;
   // double total_non_speech = total_duration - total_speech;

   double total_speech_percent = total_speech / total_duration * 100.0;

   s64 ticks = current_timestamp - stats.first_call_timestamp;

   // ticks / freq = how many ticks in 1s
   // samples in 1s = usually 16k (sample_rate global)
   // samples / 16k * freq
   s64 ticks_worth_total_processed = stats.total_samples * stats.timer_frequency;
   s64 ratio = ticks_worth_total_processed / ticks;
   double ratio_seconds = ratio / (double)HARDCODED_SAMPLE_RATE;

   int hours = (int)(total_duration / 3600.0);
   int minutes = (int)((total_duration - hours * 3600.0) / 60.0);
   int seconds = (int)(total_duration - hours * 3600.0 - minutes * 60.0);
   int milliseconds = (int)((total_duration - hours * 3600.0 - minutes * 60.0 - seconds) * 1000.0);


   if (stats.output_enabled)
   {
      fprintf(stderr, "time=%02d:%02d:%02d.%04d", hours, minutes, seconds, milliseconds);
      fprintf(stderr, " %7.2f speech (%5.1f%%), %5.1f / %5.1f (%5.1fx)\r",
              total_speech,
              total_speech_percent,
              total_duration,
              (double)ticks / stats.timer_frequency,
              ratio_seconds);
   }

   // last_call = current;
   // stats.last_call_timestamp = current.QuadPart;
#endif
}


typedef struct ArgOption ArgOption;
struct ArgOption
{
   String8 name;
   float value;
};

enum ArgOptionIndex
{
   ArgOptionIndex_MinSilence = 0,
   ArgOptionIndex_MinSpeech,
   ArgOptionIndex_Threshold,
   ArgOptionIndex_NegThresholdRelative,
   ArgOptionIndex_SpeechPad,
   ArgOptionIndex_Batch,
   ArgOptionIndex_SequenceCount,
   ArgOptionIndex_AudioSource,
   ArgOptionIndex_StartSeconds,
   ArgOptionIndex_RawProbabilities,
   ArgOptionIndex_Stats,
   ArgOptionIndex_OutputFormatCentiSeconds,
   ArgOptionIndex_Model,

   ArgOptionIndex_COUNT
};

ArgOption options[] = {
   {String8FromLiteral("--min_silence"),            200.0f  }, // NOTE(irwin): up from previous default 100.0f
   {String8FromLiteral("--min_speech"),             250.0f  },
   {String8FromLiteral("--threshold"),                0.5f  },
   {String8FromLiteral("--neg_threshold_relative"),   0.15f },
   {String8FromLiteral("--speech_pad"),              30.0f  },
   {String8FromLiteral("--batch"),                   96.0f  },
   {String8FromLiteral("--sequence_count"),        1536.0f  },
   {String8FromLiteral("--audio_source"),             0.0f  },
   {String8FromLiteral("--start_seconds"),            0.0f  },
   {String8FromLiteral("--raw_probabilities"),        0.0f  },
   {String8FromLiteral("--stats"),                    0.0f  },
   {String8FromLiteral("--output_centi_seconds"),     0.0f  },
   {String8FromLiteral("--model"),                    0.0f  },
};


int main()
{

#if 1
   MemoryArena main_arena = {0};

   size_t arena_capacity = Megabytes(32);
   u8 *base_address = malloc(arena_capacity);
   if (base_address == 0)
   {
      // TODO(irwin):
      fprintf(stderr, "Fatal: couldn't allocate required memory\n");
      return 1;
   }
   initializeMemoryArena(&main_arena, base_address, arena_capacity);

   MemoryArena *arena = &main_arena;
#else
   MemoryArena *arena = DEBUG_getDebugArena();
#endif


   float min_silence_duration_ms;
   float min_speech_duration_ms;
   float threshold;
   float neg_threshold_relative;
   float neg_threshold;
   float speech_pad_ms;

   Segment_Output_Format output_format = Segment_Output_Format_Seconds;

   String8 model_path_arg = {0};
   //const char *input_filename = "RED.s16le";
   String8 input_filename = {0};

   b32 raw_probabilities = 0;

   int arg_count_u8 = 0;
   String8 *arg_array_u8 = get_command_line_as_utf8(arena, &arg_count_u8);
   for (int arg_index = 1; arg_index < arg_count_u8; ++arg_index)
   {
      String8 arg_string = arg_array_u8[arg_index];

      // const char *arg_string_c = arg_array[arg_index];
      // String8 arg_string = String8FromCString(arg_string_c);
      b32 found_named_option = 0;

      for (int arg_option_index = 0; arg_option_index < ArgOptionIndex_COUNT; ++arg_option_index)
      {
         ArgOption *option = options + arg_option_index;
         if (String8_Equal(arg_string, option->name))
         {
            found_named_option = 1;

            if (arg_option_index == ArgOptionIndex_RawProbabilities)
            {
               // TODO(irwin): bool options
               option->value = 1.0f;
            }
            else if (arg_option_index == ArgOptionIndex_Stats)
            {
               // TODO(irwin): bool options
               option->value = 1.0f;
            }
            else if (arg_option_index == ArgOptionIndex_OutputFormatCentiSeconds)
            {
               // TODO(irwin): bool options
               option->value = 1.0f;
            }
            else if ( arg_option_index == ArgOptionIndex_Model )
            {
               int arg_value_index = arg_index + 1;
               if ( arg_value_index < arg_count_u8 )
               {
                  model_path_arg = arg_array_u8[arg_value_index];

                  option->value = 1.0f;
               }
               ++arg_index;
            }
            else
            {
               int arg_value_index = arg_index + 1;
               if (arg_value_index < arg_count_u8)
               {
                  String8 arg_value_string = arg_array_u8[arg_value_index];
                  String8 arg_value_string_null_terminated = String8ToCString(arena, arg_value_string);
                  float arg_value = (float)atof(arg_value_string_null_terminated.begin);
                  if (arg_value > 0.0f)
                  {
                     option->value = arg_value;
                  }
                  ++arg_index;
               }
            }
         }
      }

      if ( !found_named_option )
      {
         // TODO(irwin): trim quotes?
         input_filename = arg_array_u8[arg_index];
      }
   }

   min_silence_duration_ms = options[ArgOptionIndex_MinSilence].value;
   min_speech_duration_ms  = options[ArgOptionIndex_MinSpeech].value;
   threshold               = options[ArgOptionIndex_Threshold].value;
   neg_threshold_relative  = options[ArgOptionIndex_NegThresholdRelative].value;
   speech_pad_ms           = options[ArgOptionIndex_SpeechPad].value;
   raw_probabilities       = (options[ArgOptionIndex_RawProbabilities].value != 0.0f);
   if (options[ArgOptionIndex_OutputFormatCentiSeconds].value != 0.0f)
   {
      output_format = Segment_Output_Format_CentiSeconds;
   }
   b32 stats_output_enabled = (options[ArgOptionIndex_Stats].value != 0.0f);

   neg_threshold           = threshold - neg_threshold_relative;


//    if ( model_path_arg )
//    {
//       fwprintf( stderr, L"%s", model_path_arg );
//    }

   {

      // verify_input_output_count(session);

      run_inference( model_path_arg,
                    arena,
                    min_silence_duration_ms,
                    min_speech_duration_ms,
                    threshold,
                    neg_threshold,
                    speech_pad_ms,
                    options[ArgOptionIndex_SequenceCount].value,
                    raw_probabilities,
                    output_format,
                    input_filename,
                    stats_output_enabled,
                    (int)options[ArgOptionIndex_Batch].value,
                    (int)options[ArgOptionIndex_AudioSource].value,
                    options[ArgOptionIndex_StartSeconds].value);

   }


   return 0;
}

/*
sys.stdout.write("aselect='")
for i, speech_dict in enumerate(get_speech_timestamps_stdin(None, model, return_seconds=True)):
    if i:
        sys.stdout.write("+")
    sys.stdout.write("between(t,{},{})".format(speech_dict['start'], speech_dict['end']))
    #print(speech_dict['start'], speech_dict['end'])
sys.stdout.write("', asetpts=N/SR/TB")
echo aselect='
*/
