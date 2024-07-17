#pragma once
#include "utils.h"
#include "memory.h"
#include "string8.h"

#if !defined(ONNX_INFERENCE_ENABLED)
#define ONNX_INFERENCE_ENABLED 1
#endif // ONNX_INFERENCE_ENABLED

typedef struct Silero_Config Silero_Config;
struct Silero_Config
{
   b32 is_silero_v4;
   s32 batch_size_restriction;
   s32 batch_size;
   s32 input_count;

   size_t prob_shape_count;
   int64_t prob_shape[4];
   size_t prob_tensor_element_count;
};

typedef struct Tensor_Buffers Tensor_Buffers;
struct Tensor_Buffers
{
   int window_size_samples;
   float *input_samples;
   float *output;

   int lstm_count;
   float *lstm_h;
   float *lstm_c;

   float *lstm_h_out;
   float *lstm_c_out;
};

typedef struct VADC_Context VADC_Context;

// NOTE(irwin): forward declare
typedef struct Silero_Context Silero_Context;

struct VADC_Context
{
   void *backend;

   Tensor_Buffers buffers;

   b32 is_silero_v4;
   s32 silero_probability_out_index;
   const int batch_size;
};

#if ONNX_INFERENCE_ENABLED
#include "onnx_helpers.h"
#endif // ONNX_INFERENCE_ENABLED

#include <stdio.h>

#define SILERO_V4 0

// #define SILERO_FILENAME_V3 L"silero_vad_v3.onnx"
// #define SILERO_FILENAME_V3_B32 L"silero_restored_v3.1_16k_v2_b32.onnx"
#define SILERO_FILENAME_V4 L"silero_vad_v4.onnx"
#define SILERO_FILENAME_V3_B_DYNAMIC L"silero_restored_v3.1_16k_v2_dyn-sim.onnx"

#if SILERO_V4
// #define SILERO_PROBABILITY_OUT_INDEX 0
// #define SILERO_INPUT_TENSOR_COUNT 4
#define SILERO_FILENAME SILERO_FILENAME_V4
#else
// #define SILERO_PROBABILITY_OUT_INDEX 1
// #define SILERO_INPUT_TENSOR_COUNT 3
#define SILERO_FILENAME SILERO_FILENAME_V3_B_DYNAMIC
#endif

// const char *INPUT_NAMES_V4[] = { "input", "h", "c", "sr" };
// const char *INPUT_NAMES_V3[] = { "input", "h0", "c0" };
// const char *OUTPUT_NAMES_NORMAL[] = { "output", "hn", "cn" };

#define SILERO_SLICE_SAMPLES_8K  128
#define SILERO_SLICE_SAMPLES_16K 256

#define SILERO_SLICE_COUNT_MIN 2
#define SILERO_SLICE_COUNT 6

// 512, 768, 1024, 1280, 1536
#define SILERO_WINDOW_SIZE_SAMPLES (SILERO_SLICE_SAMPLES_16K * SILERO_SLICE_COUNT)

#define SILERO_SAMPLE_RATE 16000

const size_t HARDCODED_WINDOW_SIZE_SAMPLES = SILERO_WINDOW_SIZE_SAMPLES;
const size_t HARDCODED_SAMPLE_RATE = SILERO_SAMPLE_RATE;

#undef SILERO_WINDOW_SIZE_SAMPLES
#undef SILERO_SAMPLE_RATE
#undef SILERO_SLICE_COUNT_MIN
#undef SILERO_SLICE_COUNT
#undef SILERO_SLICE_SAMPLES_8K
#undef SILERO_SLICE_SAMPLES_16K


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
   Segment_Output_Format_CentiSeconds, // NOTE(irwin): hundredths of seconds, 500 -> 5 seconds

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

void process_chunks( MemoryArena *arena, VADC_Context context,
                    const size_t buffered_samples_count,
                    const float *samples_buffer_float32,
                    float *probabilities_buffer );


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
                         VADC_Stats *stats,
                         float seconds_per_chunk );

FeedProbabilityResult combine_or_emit_speech_segment( FeedProbabilityResult buffered,
                                                     FeedProbabilityResult feed_result,
                                                     float speech_pad_ms,
                                                     Segment_Output_Format output_format,
                                                     VADC_Stats *stats,
                                                     float seconds_per_chunk );

// NOTE(irwin): onnx helper routines

static inline void print_speech_stats(VADC_Stats stats);
