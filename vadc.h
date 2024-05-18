#pragma once
#include "utils.h"
#include "memory.h"
#include "string8.h"

#if !defined(ONNX_INFERENCE_ENABLED)
#define ONNX_INFERENCE_ENABLED 1
#endif // ONNX_INFERENCE_ENABLED

#if ONNX_INFERENCE_ENABLED
#include "onnx_helpers.h"
#endif // ONNX_INFERENCE_ENABLED

#include <stdio.h>

#define SILERO_V4 0

#define SILERO_FILENAME_V4 L"silero_vad_v4.onnx"
#define SILERO_FILENAME_V3 L"silero_vad_v3.onnx"
#define SILERO_FILENAME_V3_B32 L"silero_restored_v3.1_16k_v2_b32.onnx"
#define SILERO_FILENAME_V3_B_DYNAMIC L"silero_restored_v3.1_16k_v2_dyn.onnx"

#if SILERO_V4
#define SILERO_PROBABILITY_OUT_INDEX 0
#define SILERO_INPUT_TENSOR_COUNT 4
#define SILERO_FILENAME L"silero_vad_v4.onnx"
#else
#define SILERO_PROBABILITY_OUT_INDEX 1
#define SILERO_INPUT_TENSOR_COUNT 3
#define SILERO_FILENAME SILERO_FILENAME_V3_B_DYNAMIC
#endif

#define SILERO_WINDOW_SIZE_SAMPLES 1536
#define SILERO_SAMPLE_RATE 16000

// TODO(irwin): fix, globals shouldn't be lowercase like that
const size_t window_size_samples = SILERO_WINDOW_SIZE_SAMPLES;
const size_t sample_rate = SILERO_SAMPLE_RATE;
const float chunks_per_second = (float)SILERO_SAMPLE_RATE / SILERO_WINDOW_SIZE_SAMPLES;

const float chunk_duration_ms = SILERO_WINDOW_SIZE_SAMPLES / (float)SILERO_SAMPLE_RATE * 1000.0f;


typedef struct VADC_Context VADC_Context;

// NOTE(irwin): forward declare
typedef struct Silero_Context Silero_Context;

struct VADC_Context
{
#if ONNX_INFERENCE_ENABLED
   const OrtValue *const *input_tensors;
   OrtValue **output_tensors;
   OrtSession *session;
#endif
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
   const int batch_size;

   Silero_Context *silero_context;
};

typedef struct VADC_Chunk_Result
{
   float probability;

   // NOTE(irwin): unused, remove
   size_t state_count;
   //float *state_h;
   //float *state_c;
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

typedef struct VADC_Stats VADC_Stats;
struct VADC_Stats
{
   s64 first_call_timestamp;
   // s64 last_call_timestamp;
   s64 timer_frequency;

   double total_speech;
   double total_duration;
   s64 total_samples;

   b32 output_enabled;
};

typedef enum Segment_Output_Format Segment_Output_Format;
enum Segment_Output_Format
{
   Segment_Output_Format_Seconds = 0,
   Segment_Output_Format_CentiSeconds, // NOTE(irwin): hundreths of seconds, 500 -> 5 seconds

   Segment_Output_Format_COUNT
};


int run_inference( String8 model_path_arg,
                  MemoryArena *arena,
                  float min_silence_duration_ms,
                  float min_speech_duration_ms,
                  float threshold,
                  float neg_threshold,
                  float speech_pad_ms,
                  b32 raw_probabilities,
                  Segment_Output_Format output_format,
                  String8 filename,
                  b32 stats_output_enabled,
                  s32 preferred_batch_size,
                  int audio_source,
                  float start_seconds );

void process_chunks( VADC_Context context,
                    const size_t buffered_samples_count,
                    const float *samples_buffer_float32,
                    float *probabilities_buffer );

float run_inference_on_single_chunk( VADC_Context context,
                                                const size_t samples_count,
                                                const float *samples_buffer_float32,
                                                float *state_h_in,
                                                float *state_c_in );

FeedProbabilityResult feed_probability( FeedState *state,
                                       int min_silence_duration_chunks,
                                       int min_speech_duration_chunks,
                                       float probability,
                                       float threshold,
                                       float neg_threshold,
                                       int global_chunk_index );

void emit_speech_segment( FeedProbabilityResult segment,
                         float speech_pad_ms,
                         Segment_Output_Format output_format,
                         VADC_Stats *stats );

FeedProbabilityResult combine_or_emit_speech_segment( FeedProbabilityResult buffered,
                                                     FeedProbabilityResult feed_result,
                                                     float speech_pad_ms,
                                                     Segment_Output_Format output_format,
                                                     VADC_Stats *stats );

// NOTE(irwin): onnx helper routines

static inline void print_speech_stats(VADC_Stats stats);
