#include "vadc.h"

#include <inttypes.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h> // GetModuleFileNameW
#include <Shlwapi.h> // PathRemoveFileSpecW, PathAppendW

#include "onnx_helpers.c"

#include "string8.c"

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


static const wchar_t model_filename[] = SILERO_FILENAME;


VADC_Chunk_Result run_inference_on_single_chunk( VADC_Context context,
                                                 const size_t samples_count,
                                                 const float *samples_buffer_float32,
                                                 float *state_h_in,
                                                 float *state_c_in )
{
   Assert( samples_count > 0 && samples_count <= context.window_size_samples );
#if DEBUG_WRITE_STATE_TO_FILE
   FILE *debug_file = getDebugFile();
#endif

   VADC_Chunk_Result result = { 0 };

   memmove( context.input_tensor_samples, samples_buffer_float32, samples_count * sizeof( context.input_tensor_samples[0] ) );

   // NOTE(irwin): pad chunks with not enough samples
   if ( samples_count < context.window_size_samples )
   {
      for ( size_t pad_index = samples_count; pad_index < context.window_size_samples; ++pad_index )
      {
         context.input_tensor_samples[pad_index] = 0.0f;
      }
   }

   memmove( context.input_tensor_state_h, state_h_in, context.state_count * sizeof( context.input_tensor_state_h[0] ) );
   memmove( context.input_tensor_state_c, state_c_in, context.state_count * sizeof( context.input_tensor_state_c[0] ) );
#if DEBUG_WRITE_STATE_TO_FILE
   DEBUG_Silero_State debug_state;
   float *source = context.input_tensor_samples;
   size_t source_size_bytes = 1536 * sizeof( context.input_tensor_samples[0] );
   memmove( debug_state.samples, source, source_size_bytes );
   memmove( debug_state.state_h, context.input_tensor_state_h, context.state_count * sizeof( context.input_tensor_state_h[0] ) );
   memmove( debug_state.state_c, context.input_tensor_state_c, context.state_count * sizeof( context.input_tensor_state_c[0] ) );
   fwrite( &debug_state, sizeof( DEBUG_Silero_State ), 1, debug_file );
#endif

   ORT_ABORT_ON_ERROR( g_ort->Run( context.session,
                                   NULL,
                                   context.input_names,
                                   context.input_tensors,
                                   context.inputs_count,
                                   context.output_names,
                                   context.outputs_count,
                                   context.output_tensors )
   );
   Assert( context.output_tensors[0] != NULL );

   int is_tensor;
   ORT_ABORT_ON_ERROR( g_ort->IsTensor( context.output_tensors[0], &is_tensor ) );
   Assert( is_tensor );

   float result_probability = context.output_tensor_prob[context.silero_probability_out_index];
   result.probability = result_probability;

   result.state_h = context.output_tensor_state_h;
   result.state_c = context.output_tensor_state_c;

   return result;
}


void process_chunks( VADC_Context context,
                    const size_t buffered_samples_count,
                    const float *samples_buffer_float32,
                    float *probabilities_buffer)
{
   for (size_t offset = 0;
        offset < buffered_samples_count;
        offset += context.window_size_samples)
   {
      // NOTE(irwin): copy a slice of the buffered samples
      size_t samples_count_left = buffered_samples_count - offset;
      size_t window_size = samples_count_left > context.window_size_samples ? context.window_size_samples : samples_count_left;

      // IMPORTANT(irwin): hardcoded to use state from previous inference, assumed to be in output tensor memory
      // TODO(irwin): dehardcode
      VADC_Chunk_Result result = run_inference_on_single_chunk( context,
                                                                window_size,
                                                                samples_buffer_float32 + offset,
                                                                context.output_tensor_state_h,
                                                                context.output_tensor_state_c );

      *probabilities_buffer++ = result.probability;
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

static inline float to_s(int in_chunks)
{
   return in_chunks / chunks_per_second;
}

void emit_speech_segment(FeedProbabilityResult segment,
                         float speech_pad_ms,
                         Segment_Output_Format output_format,
                         VADC_Stats *stats)
{
   const float speech_pad_s = speech_pad_ms / 1000.0f;

   float speech_end_padded = to_s(segment.speech_end) + speech_pad_s;

   // NOTE(irwin): print previous start/end times padded in seconds
   float speech_start_padded = to_s(segment.speech_start) - speech_pad_s;
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
}

FeedProbabilityResult combine_or_emit_speech_segment(FeedProbabilityResult buffered, FeedProbabilityResult feed_result,
                                                     float speech_pad_ms, Segment_Output_Format output_format, VADC_Stats *stats)
{
   FeedProbabilityResult result = buffered;

   const float speech_pad_s = speech_pad_ms / 1000.0f;

   float current_speech_start_padded = to_s(feed_result.speech_start) - speech_pad_s;
   if (current_speech_start_padded < 0.0f)
   {
      current_speech_start_padded = 0.0f;
   }

   if (result.is_valid)
   {
      float buffered_speech_end_padded = to_s(result.speech_end) + speech_pad_s;
      if (buffered_speech_end_padded >= current_speech_start_padded)
      {
         result.speech_end = feed_result.speech_end;
      }
      else
      {
         emit_speech_segment(result, speech_pad_ms, output_format, stats);

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

static void init_buffered_stream_ffmpeg(MemoryArena *arena, Buffered_Stream *s, String8 fname_inp, size_t buffer_size)
{
   memset( s, 0, sizeof( *s ) );

   const char *ffmpeg_to_s16le = "ffmpeg -hide_banner -loglevel error -stats -i \"%.*s\" -map 0:a:0 -vn -sn -dn -ac 1 -ar 16k -f s16le -";
   String8 ffmpeg_command = String8_pushf(arena, ffmpeg_to_s16le, fname_inp.size, fname_inp.begin);
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


int run_inference(OrtSession* session,
                  MemoryArena *arena,
                  float min_silence_duration_ms,
                  float min_speech_duration_ms,
                  float threshold,
                  float neg_threshold,
                  float speech_pad_ms,
                  b32 raw_probabilities,
                  Segment_Output_Format output_format,
                  String8 filename ) {
   size_t model_input_count = 0;
   ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount( session, &model_input_count ));
   Assert( model_input_count == 3 || model_input_count == 4 );
   const b32 is_silero_v4 = model_input_count == 4;

   OrtMemoryInfo* memory_info;
   ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
   OrtAllocator* ort_allocator;
   ORT_ABORT_ON_ERROR(g_ort->CreateAllocator(session, memory_info, &ort_allocator));

   size_t model_output_count = 0;
   ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount( session, &model_output_count ));

   // for (int i = 0; i < model_output_count; ++i)
   // {
   //    char *output_name = 0;
   //    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, ort_allocator, &output_name));
   //    fprintf(stderr, "output name: %s\n", output_name);
   // }

   // const float threshold               = 0.5f;
   // const float neg_threshold           = threshold - 0.15f;
   // const float min_speech_duration_ms  = 250.0f;
   // const float min_silence_duration_ms = 100.0f;
   // const float speech_pad_ms           = 30.0f;

   int min_speech_duration_chunks = (int)(min_speech_duration_ms / chunk_duration_ms + 0.5f);
   if (min_speech_duration_chunks < 1)
   {
      min_speech_duration_chunks = 1;
   }

   int min_silence_duration_chunks = (int)(min_silence_duration_ms / chunk_duration_ms + 0.5f);
   if (min_silence_duration_chunks < 1)
   {
      min_silence_duration_chunks = 1;
   }


   // NOTE(irwin): create tensors and allocate tensors backing memory buffers
   const size_t input_count = window_size_samples;
   // float *input_tensor_samples = (float *)malloc(input_count * sizeof(float));
   float *input_tensor_samples = pushArray(arena, input_count, float);

   int64_t input_tensor_samples_shape[] = {1, input_count};
   const size_t input_tensor_samples_shape_count = ArrayCount(input_tensor_samples_shape);

   const size_t silero_input_tensor_count = is_silero_v4 ? 4 : 3;
   OrtValue* input_tensors[4];

   create_tensor(memory_info, &input_tensors[0], input_tensor_samples_shape, input_tensor_samples_shape_count, input_tensor_samples, input_count);

   const size_t state_count = 128;
   // float *state_h = (float *)malloc(state_count * sizeof(float));
   float *state_h = pushArray(arena, state_count, float);
   memset(state_h, 0, state_count * sizeof(float));

   // float *state_c = (float *)malloc(state_count * sizeof(float));
   float *state_c = pushArray(arena, state_count, float);
   memset(state_c, 0, state_count * sizeof(float));

   // float *state_h_out = (float *)malloc(state_count * sizeof(float));
   float *state_h_out = pushArray(arena, state_count, float);
   memset(state_h_out, 0, state_count * sizeof(float));

   // float *state_c_out = (float *)malloc(state_count * sizeof(float));
   float *state_c_out = pushArray(arena, state_count, float);
   memset(state_c_out, 0, state_count * sizeof(float));

   int64_t state_shape[] = {2, 1, 64};
   const size_t state_shape_count = ArrayCount(state_shape);

   OrtValue** state_h_tensor = &input_tensors[1];
   create_tensor(memory_info, state_h_tensor, state_shape, state_shape_count, state_h, state_count);

   OrtValue** state_c_tensor = &input_tensors[2];
   create_tensor(memory_info, state_c_tensor, state_shape, state_shape_count, state_c, state_count);

   if ( is_silero_v4 )
   {
      int64_t sr = 16000;
      int64_t sr_shape[] = { 1, 1 };
      const size_t sr_shape_count = ArrayCount( sr_shape );
      OrtValue **sr_tensor = &input_tensors[3];
      create_tensor_int64( memory_info, sr_tensor, sr_shape, sr_shape_count, &sr, 1 );
   }

   const char *input_names_v4[] = { "input", "h", "c", "sr" };
   const char *input_names_v3[] = { "input", "h0", "c0" };

   VAR_UNUSED( input_names_v4 );
   VAR_UNUSED( input_names_v3 );

   int64_t prob_shape_v4[] = { 1, 1 };
   int64_t prob_shape_v3[] = { 1, 2, 1 };

   VAR_UNUSED( prob_shape_v4 );
   VAR_UNUSED( prob_shape_v3 );

   const size_t prob_shape_count_v4 = ArrayCount( prob_shape_v4 );
   const size_t prob_shape_count_v3 = ArrayCount( prob_shape_v3 );
   VAR_UNUSED( prob_shape_count_v4 );
   VAR_UNUSED( prob_shape_count_v3 );

   const char **input_names = is_silero_v4 ? input_names_v4 : input_names_v3;
   int64_t *prob_shape = is_silero_v4 ? prob_shape_v4 : prob_shape_v3;

   float prob[2];

   const char *output_names[] = { "output", "hn", "cn" };
   OrtValue *output_tensors[3] = { 0 };
   OrtValue **output_prob_tensor = &output_tensors[0];

   const size_t prob_shape_count = is_silero_v4 ? prob_shape_count_v4 : prob_shape_count_v3;

   create_tensor(memory_info, output_prob_tensor, prob_shape, prob_shape_count, &prob[0], prob_shape_count);

   OrtValue** state_h_out_tensor = &output_tensors[1];
   create_tensor(memory_info, state_h_out_tensor, state_shape, state_shape_count, state_h_out, state_count);

   OrtValue** state_c_out_tensor = &output_tensors[2];
   create_tensor(memory_info, state_c_out_tensor, state_shape, state_shape_count, state_c_out, state_count);

   // g_ort->ReleaseMemoryInfo(memory_info);


   // NOTE(irwin): read samples from a file or stdin and run inference
   // NOTE(irwin): at 16000 sampling rate, one chunk is 96 ms or 1536 samples
   // NOTE(irwin): chunks count being 96, the same as one chunk's length in milliseconds,
   // is purely coincidental
   const int chunks_count = 96;
   // NOTE(irwin): buffered_samples_count is the normalization window size
   const size_t buffered_samples_count = window_size_samples * chunks_count;

   // short *samples_buffer_s16 = (short *)malloc(buffered_samples_count * sizeof(short));
   short *samples_buffer_s16 = pushArray(arena, buffered_samples_count, short);
   // float *samples_buffer_float32 = (float *)malloc(buffered_samples_count * sizeof(float));
   float *samples_buffer_float32 = pushArray(arena, buffered_samples_count, float);
   // float *probabilities_buffer = (float *)malloc(chunks_count * sizeof(float));
   float *probabilities_buffer = pushArray(arena, chunks_count, float);

   Buffered_Stream read_stream = {0};

   size_t buffered_samples_size_in_bytes = sizeof( short ) * buffered_samples_count;
   if (filename.size)
   {
      init_buffered_stream_ffmpeg(arena, &read_stream, filename, buffered_samples_size_in_bytes );
   }
   else
   {
      init_buffered_stream_stdin(arena, &read_stream, buffered_samples_size_in_bytes );
   }



   VADC_Context context =
   {
      .input_tensors = input_tensors,
      .output_tensors = output_tensors,
      .session = session,
      .input_names = input_names,
      .output_names = output_names,
      .state_count = state_count,
      .input_tensor_state_h = state_h,
      .input_tensor_state_c = state_c,
      .output_tensor_state_h = state_h_out,
      .output_tensor_state_c = state_c_out,
      .window_size_samples = window_size_samples,
      .output_tensor_prob = prob,
      .input_tensor_samples = input_tensor_samples,

      .inputs_count = silero_input_tensor_count,
      .outputs_count = 3,
      .is_silero_v4 = is_silero_v4,
      .silero_probability_out_index = is_silero_v4 ? 0 : 1
   };

   FeedState state = {0};
   int global_chunk_index = 0;

   FeedProbabilityResult buffered = {0};

   VADC_Stats stats = {0};

   s64 total_samples_read = 0;
   size_t values_read = 0;
   for(;;)
   {
      BS_Error read_error_code = 0;

      // TODO(irwin): what do we do about errors that arose in refilling the buffered stream
      // but some data was still read? Like EOF, or closed pipe?

      read_error_code = read_stream.refill( &read_stream );

      values_read = (read_stream.end - read_stream.start) / sizeof(short);
      total_samples_read += values_read;
      stats.total_duration = (double)total_samples_read / sample_rate;

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

         if (max_value > 0.0f)
         {
            for (size_t i = 0; i < values_read; ++i)
            {
               samples_buffer_float32[i] /= max_value;
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

      process_chunks( context,
                     values_read,
                     samples_buffer_float32,
                     probabilities_buffer);

      int probabilities_count = (int)(values_read / window_size_samples);
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
                                                      speech_pad_ms, output_format, &stats);
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
         int audio_length_samples = (int)((global_chunk_index - 1) * window_size_samples);
         if (audio_length_samples - (state.current_speech_start * window_size_samples) > (min_speech_duration_chunks * window_size_samples))
         {
            FeedProbabilityResult final_segment;
            final_segment.is_valid = 1;
            final_segment.speech_start = state.current_speech_start;
            final_segment.speech_end = (int)(audio_length_samples / window_size_samples);

            buffered = combine_or_emit_speech_segment(buffered, final_segment,
                                                         speech_pad_ms, output_format, &stats);
         }
      }

      if (buffered.is_valid)
      {
         emit_speech_segment(buffered, speech_pad_ms, output_format, &stats);
      }
   }

   print_speech_stats(stats);

   // g_ort->ReleaseValue(output_tensor);
   // g_ort->ReleaseValue(input_tensor);
   // return ret;
   return 0;
}

void print_speech_stats(VADC_Stats stats)
{
   double total_speech = stats.total_speech;
   double total_duration = stats.total_duration;
   double total_non_speech = total_duration - total_speech;

   double total_speech_percent = total_speech / total_duration * 100.0;

#if 1
   fprintf(stderr, "%.2f speech, %.2f non-speech, (%.1f%% total speech)\n", total_speech, total_non_speech, total_speech_percent);
#else
   fprintf(stderr, "%f total speech\n", total_speech);
   fprintf(stderr, "%f total non-speech\n", total_non_speech);
   fprintf(stderr, "%f total percentage of speech\n", total_speech_percent);
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
   ArgOptionIndex_RawProbabilities,
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
   {String8FromLiteral("--raw_probabilities"),        0.0f  },
   {String8FromLiteral("--output_centi_seconds"),     0.0f  },
   {String8FromLiteral("--model"),                    0.0f  },
};

int main()
{
   MemoryArena arena = {0};
   size_t arena_capacity = Megabytes(32);
   u8 *base_address = malloc(arena_capacity);
   if (base_address == 0)
   {
      // TODO(irwin):
      fprintf(stderr, "Fatal: couldn't allocate required memory\n");
      return 1;
   }
   initializeMemoryArena(&arena, base_address, arena_capacity);


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
   String8 *arg_array_u8 = get_command_line_as_utf8(&arena, &arg_count_u8);
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
            }
            else
            {
               int arg_value_index = arg_index + 1;
               if (arg_value_index < arg_count_u8)
               {
                  String8 arg_value_string = arg_array_u8[arg_value_index];
                  String8 arg_value_string_null_terminated = String8ToCString(&arena, arg_value_string);
                  float arg_value = (float)atof(arg_value_string_null_terminated.begin);
                  if (arg_value > 0.0f)
                  {
                     option->value = arg_value;
                  }
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

   neg_threshold           = threshold - neg_threshold_relative;

   g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
   if (!g_ort)
   {
      fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
      return -1;
   }

   OrtEnv* env;
   ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &env));
   Assert(env != NULL);

   OrtSessionOptions* session_options;
   ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
   // enable_cuda(session_options);
   ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, 1));
   ORT_ABORT_ON_ERROR(g_ort->SetInterOpNumThreads(session_options, 1));

#define MODEL_PATH_BUFFER_SIZE 1024
   wchar_t *model_path_arg_w = 0;
   String8_ToWidechar(&arena, &model_path_arg_w, model_path_arg);

   const size_t model_path_buffer_size = MODEL_PATH_BUFFER_SIZE;
   wchar_t model_path[MODEL_PATH_BUFFER_SIZE];
   GetModuleFileNameW(NULL, model_path, (DWORD)model_path_buffer_size);
   PathRemoveFileSpecW( model_path );
   PathAppendW( model_path, model_path_arg_w ? model_path_arg_w : model_filename );

//    if ( model_path_arg )
//    {
//       fwprintf( stderr, L"%s", model_path_arg );
//    }

   {
      OrtSession* session;
      ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

      // verify_input_output_count(session);

      run_inference(session,
                    &arena,
                    min_silence_duration_ms,
                    min_speech_duration_ms,
                    threshold,
                    neg_threshold,
                    speech_pad_ms,
                    raw_probabilities,
                    output_format, input_filename);

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
