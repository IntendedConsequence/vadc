#pragma once
#include "utils.h"

#include <stdio.h>
#include "include/onnxruntime_c_api.h"

#define SILERO_V4 0

#define SILERO_FILENAME_V4 L"silero_vad_v4.onnx"
#define SILERO_FILENAME_V3 L"silero_vad_v3.onnx"

#if SILERO_V4
#define SILERO_PROBABILITY_OUT_INDEX 0
#define SILERO_INPUT_TENSOR_COUNT 4
#define SILERO_FILENAME L"silero_vad_v4.onnx"
#else
#define SILERO_PROBABILITY_OUT_INDEX 1
#define SILERO_INPUT_TENSOR_COUNT 3
#define SILERO_FILENAME L"silero_vad_v3.onnx"
#endif

#define SILERO_WINDOW_SIZE_SAMPLES 1536
#define SILERO_SAMPLE_RATE 16000

const size_t window_size_samples = SILERO_WINDOW_SIZE_SAMPLES;
const size_t sample_rate = SILERO_SAMPLE_RATE;
const float chunks_per_second = (float)SILERO_SAMPLE_RATE / SILERO_WINDOW_SIZE_SAMPLES;

const float chunk_duration_ms = SILERO_WINDOW_SIZE_SAMPLES / (float)SILERO_SAMPLE_RATE * 1000.0f;


#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

typedef struct VADC_Context
{
   const OrtValue *const *input_tensors;
   OrtValue **output_tensors;
   OrtSession *session;
   const char **input_names;
   const char **output_names;
   const size_t inputs_count;
   const size_t outputs_count;

   float *output_tensor_state_h;
   float *output_tensor_state_c;
   float *output_tensor_prob;

   const size_t window_size_samples;
   float *input_tensor_samples;

   const size_t state_count;
   float *input_tensor_state_h;
   float *input_tensor_state_c;
   b32 is_silero_v4;
   s32 silero_probability_out_index;
} VADC_Context;

typedef struct VADC_Chunk_Result
{
   float probability;

   size_t state_count;
   float *state_h;
   float *state_c;
} VADC_Chunk_Result;

typedef struct FeedState
{
   int temp_end;
   int current_speech_start;
   b32 triggered;
} FeedState;

typedef struct FeedProbabilityResult
{
   int speech_start;
   int speech_end;
   b32 is_valid;
} FeedProbabilityResult;



int run_inference( OrtSession *session, float min_silence_duration_ms, float min_speech_duration_ms, float threshold, float neg_threshold, float speech_pad_ms, b32 raw_probabilities );
void process_chunks( VADC_Context context, const size_t buffered_samples_count, const float *samples_buffer_float32, float *probabilities_buffer );
VADC_Chunk_Result run_inference_on_single_chunk( VADC_Context context, const size_t samples_count, const float *samples_buffer_float32, float *state_h_in, float *state_c_in );

FeedProbabilityResult feed_probability( FeedState *state, int min_silence_duration_chunks, int min_speech_duration_chunks, float probability, float threshold, float neg_threshold, int global_chunk_index );
void emit_speech_segment( FeedProbabilityResult segment, float speech_pad_ms );
FeedProbabilityResult combine_or_emit_speech_segment( FeedProbabilityResult buffered, FeedProbabilityResult feed_result, float speech_pad_ms );

// NOTE(irwin): onnx helper routines
void verify_input_output_count( OrtSession *session );
void create_tensor( OrtMemoryInfo *memory_info, OrtValue **out_tensor, int64_t *shape, size_t shape_count, float *input, size_t input_count );
void create_tensor_int64( OrtMemoryInfo *memory_info, OrtValue **out_tensor, int64_t *shape, size_t shape_count, int64_t *input, size_t input_count );
int enable_cuda( OrtSessionOptions *session_options );
