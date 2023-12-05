#include "vadc.h"

#include <assert.h>
#include <io.h>
#include <fcntl.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <Shlwapi.h>

#include "onnx_helpers.c"
#ifndef FROM_STDIN
#define FROM_STDIN 1
#endif

#ifndef DEBUG_WRITE_STATE_TO_FILE
#define DEBUG_WRITE_STATE_TO_FILE 0
#endif

#if DEBUG_WRITE_STATE_TO_FILE
typedef struct DEBUG_Silero_State 
{
   float samples[1536];
   float state_h[128];
   float state_c[128];
} DEBUG_Silero_State;

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
   assert( context.output_tensors[0] != NULL );

   int is_tensor;
   ORT_ABORT_ON_ERROR( g_ort->IsTensor( context.output_tensors[0], &is_tensor ) );
   assert( is_tensor );

   float result_probability = context.output_tensor_prob[SILERO_PROBABILITY_OUT_INDEX];
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

void emit_speech_segment(FeedProbabilityResult segment, float speech_pad_ms)
{
   const float speech_pad_s = speech_pad_ms / 1000.0f;

   float speech_end_padded = to_s(segment.speech_end) + speech_pad_s;

   // NOTE(irwin): print previous start/end times padded in seconds
   float speech_start_padded = to_s(segment.speech_start) - speech_pad_s;
   if (speech_start_padded < 0.0f)
   {
      speech_start_padded = 0.0f;
   }

   fprintf(stdout, "%.2f,%.2f\n", speech_start_padded, speech_end_padded);
   fflush(stdout);
}

FeedProbabilityResult combine_or_emit_speech_segment(FeedProbabilityResult buffered, FeedProbabilityResult feed_result,
                                                     float speech_pad_ms)
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
         emit_speech_segment(result, speech_pad_ms);

         result = feed_result;
      }
   }
   else
   {
      result = feed_result;
   }

   return result;
}

int run_inference(OrtSession* session,
                  float min_silence_duration_ms,
                  float min_speech_duration_ms,
                  float threshold,
                  float neg_threshold,
                  float speech_pad_ms,
                  b32 raw_probabilities) {
   OrtMemoryInfo* memory_info;
   ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

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
   float *input_tensor_samples = (float *)malloc(input_count * sizeof(float));

   int64_t input_tensor_samples_shape[] = {1, input_count};
   const size_t input_tensor_samples_shape_count = ArrayCount(input_tensor_samples_shape);

   OrtValue* input_tensors[SILERO_INPUT_TENSOR_COUNT];

   create_tensor(memory_info, &input_tensors[0], input_tensor_samples_shape, input_tensor_samples_shape_count, input_tensor_samples, input_count);

   const size_t state_count = 128;
   float *state_h = (float *)malloc(state_count * sizeof(float));
   memset(state_h, 0, state_count * sizeof(float));

   float *state_c = (float *)malloc(state_count * sizeof(float));
   memset(state_c, 0, state_count * sizeof(float));

   float *state_h_out = (float *)malloc(state_count * sizeof(float));
   memset(state_h_out, 0, state_count * sizeof(float));

   float *state_c_out = (float *)malloc(state_count * sizeof(float));
   memset(state_c_out, 0, state_count * sizeof(float));

   int64_t state_shape[] = {2, 1, 64};
   const size_t state_shape_count = ArrayCount(state_shape);

   OrtValue** state_h_tensor = &input_tensors[1];
   create_tensor(memory_info, state_h_tensor, state_shape, state_shape_count, state_h, state_count);

   OrtValue** state_c_tensor = &input_tensors[2];
   create_tensor(memory_info, state_c_tensor, state_shape, state_shape_count, state_c, state_count);

   if ( SILERO_V4 )
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

   const char **input_names = SILERO_V4 ? input_names_v4 : input_names_v3;
   int64_t *prob_shape = SILERO_V4 ? prob_shape_v4 : prob_shape_v3;

   float prob[2];

   const char *output_names[] = { "output", "hn", "cn" };
   OrtValue *output_tensors[3] = { 0 };
   OrtValue **output_prob_tensor = &output_tensors[0];
   
   const size_t prob_shape_count = SILERO_V4 ? prob_shape_count_v4 : prob_shape_count_v3;

   create_tensor(memory_info, output_prob_tensor, prob_shape, prob_shape_count, &prob[0], prob_shape_count);

   OrtValue** state_h_out_tensor = &output_tensors[1];
   create_tensor(memory_info, state_h_out_tensor, state_shape, state_shape_count, state_h_out, state_count);

   OrtValue** state_c_out_tensor = &output_tensors[2];
   create_tensor(memory_info, state_c_out_tensor, state_shape, state_shape_count, state_c_out, state_count);

   // g_ort->ReleaseMemoryInfo(memory_info);


   // NOTE(irwin): read samples from a file or stdin and run inference
   // NOTE(irwin): at 16000 sampling rate, one chunk is 96 ms or 1536 samples
   const int chunks_count = 96;
   const size_t buffered_samples_count = window_size_samples * chunks_count;

   short *samples_buffer_s16 = (short *)malloc(buffered_samples_count * sizeof(short));
   float *samples_buffer_float32 = (float *)malloc(buffered_samples_count * sizeof(float));
   float *probabilities_buffer = (float *)malloc(chunks_count * sizeof(float));

   FILE* read_source;
#if FROM_STDIN

#ifdef __clang__
   #pragma clang diagnostic push
   #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif //__clang__

   _setmode(_fileno(stdin), O_BINARY);

#ifdef __clang__
   #pragma clang diagnostic pop
#endif //__clang__

   read_source = stdin;
#else
   read_source = fopen("RED.s16le", "rb");
#endif


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

      .inputs_count = SILERO_INPUT_TENSOR_COUNT,
      .outputs_count = 3
   };

   FeedState state = {0};
   int global_chunk_index = 0;

   FeedProbabilityResult buffered = {0};


   size_t values_read = 0;
   for(;;)
   {
      values_read = fread(samples_buffer_s16, sizeof(short), buffered_samples_count, read_source);
      // fprintf(stderr, "%zu\n", values_read);

      if (values_read > 0)
      {
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

         for (size_t i = 0; i < values_read; ++i)
         {
            samples_buffer_float32[i] /= max_value;
         }
      }
      else
      {
         // if (feof(read_source))
         // {
         //    puts("EOF indicator set");
         // }
         // if (ferror(read_source))
         // {
         //    puts("Error indicator set");
         // }
         break;
      }

      process_chunks( context,
                     values_read,
                     samples_buffer_float32,
                     probabilities_buffer);

      if (!raw_probabilities)
      {
         for (size_t i = 0; i < values_read / window_size_samples; ++i)
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
                                                      speech_pad_ms);
         }

            // printf("%f\n", probability);
            ++global_chunk_index;
         }
      }
      else
      {
         for (size_t i = 0; i < values_read / window_size_samples; ++i)
         {
            float probability = probabilities_buffer[i];
            printf("%f\n", probability);
            ++global_chunk_index;
         }
      }

   }

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
                                                      speech_pad_ms);
      }
   }

   if (buffered.is_valid)
      {
         emit_speech_segment(buffered, speech_pad_ms);
      }
   }

   // g_ort->ReleaseValue(output_tensor);
   // g_ort->ReleaseValue(input_tensor);
   // return ret;
   return 0;
}


typedef struct ArgOption
{
   const char *name;
   float value;
} ArgOption;

enum ArgOptionIndex
{
   ArgOptionIndex_MinSilence = 0,
   ArgOptionIndex_MinSpeech,
   ArgOptionIndex_Threshold,
   ArgOptionIndex_NegThresholdRelative,
   ArgOptionIndex_SpeechPad,
   ArgOptionIndex_RawProbabilities,

   ArgOptionIndex_COUNT
};

ArgOption options[] = {
   {"--min_silence", 200.0f}, // NOTE(irwin): up from previous default 100.0f
   {"--min_speech", 250.0f},
   {"--threshold", 0.5f},
   {"--neg_threshold_relative", 0.15f},
   {"--speech_pad", 30.0f},
   {"--raw_probabilities", 0.0f},
};

int main(int arg_count, char **arg_array)
{
   float min_silence_duration_ms;
   float min_speech_duration_ms;
   float threshold;
   float neg_threshold_relative;
   float neg_threshold;
   float speech_pad_ms;

   b32 raw_probabilities = 0;

   for (int arg_index = 1; arg_index < arg_count; ++arg_index)
   {
      const char *arg_string = arg_array[arg_index];

      for (int arg_option_index = 0; arg_option_index < ArgOptionIndex_COUNT; ++arg_option_index)
      {
         ArgOption *option = options + arg_option_index;
         if (strcmp(arg_string, option->name) == 0)
         {
            if (arg_option_index == ArgOptionIndex_RawProbabilities)
            {
               // TODO(irwin): bool options
               option->value = 1.0f;
            }
            else
            {
               int arg_value_index = arg_index + 1;
               if (arg_value_index < arg_count)
               {
                  const char *arg_value_string = arg_array[arg_value_index];
                  float arg_value = (float)atof(arg_value_string);
                  if (arg_value > 0.0f)
                  {
                     option->value = arg_value;
                  }
               }
            }
         }
      }
   }

   min_silence_duration_ms = options[ArgOptionIndex_MinSilence].value;
   min_speech_duration_ms  = options[ArgOptionIndex_MinSpeech].value;
   threshold               = options[ArgOptionIndex_Threshold].value;
   neg_threshold_relative  = options[ArgOptionIndex_NegThresholdRelative].value;
   speech_pad_ms           = options[ArgOptionIndex_SpeechPad].value;
   raw_probabilities       = (options[ArgOptionIndex_RawProbabilities].value != 0.0f);

   neg_threshold           = threshold - neg_threshold_relative;

   g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
   if (!g_ort)
   {
      fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
      return -1;
   }

   OrtEnv* env;
   ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &env));
   assert(env != NULL);

   OrtSessionOptions* session_options;
   ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
   // enable_cuda(session_options);
   ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, 1));
   ORT_ABORT_ON_ERROR(g_ort->SetInterOpNumThreads(session_options, 1));

#define MODEL_PATH_BUFFER_SIZE 1024
   const size_t model_path_buffer_size = MODEL_PATH_BUFFER_SIZE;
   wchar_t model_path[MODEL_PATH_BUFFER_SIZE];
   GetModuleFileNameW(NULL, model_path, (DWORD)model_path_buffer_size);
   PathRemoveFileSpecW( model_path );
   PathAppendW( model_path, model_filename );

   {
      OrtSession* session;
      ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

      // verify_input_output_count(session);

      run_inference(session,
                    min_silence_duration_ms,
                    min_speech_duration_ms,
                    threshold,
                    neg_threshold,
                    speech_pad_ms,
                    raw_probabilities);

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
