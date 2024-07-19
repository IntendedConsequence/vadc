#pragma once
#include "include/onnxruntime_c_api.h"


#define ORT_ABORT_ON_ERROR(expr)                                \
   do {                                                         \
      OrtStatus* onnx_status = (expr);                          \
      if (onnx_status != NULL) {                                \
         const char* msg = g_ort->GetErrorMessage(onnx_status); \
         fprintf(stderr, "%s\n", msg);                          \
         g_ort->ReleaseStatus(onnx_status);                     \
         abort();                                               \
      }                                                         \
   } while (0);

typedef struct ONNX_Specific ONNX_Specific;
struct ONNX_Specific
{
   OrtValue *input_tensors[4];
   OrtValue *output_tensors[3];
   OrtSession *session;
   OrtMemoryInfo *memory_info;
   OrtAllocator *ort_allocator;
   const char *input_names[4];
   const char *output_names[4];
   size_t inputs_count;
   size_t outputs_count;
   // s32 batch_size_restriction;
   // b32 is_silero_v4;
};


void verify_input_output_count( OrtSession *session );

void create_tensor( OrtMemoryInfo *memory_info,
                    OrtValue **out_tensor,
                    int64_t *shape,
                    size_t shape_count,
                    float *input,
                    size_t input_count );

void create_tensor_int64( OrtMemoryInfo *memory_info,
                          OrtValue **out_tensor,
                          int64_t *shape,
                          size_t shape_count,
                          int64_t *input,
                          size_t input_count );

int enable_cuda( OrtSessionOptions *session_options );

static void *ort_init( MemoryArena * arena, String8 model_path_arg, s32 *batch_size_restriction, s32 *sr_input_index, s32 *output_dims, s32 *lstm_hidden_size);
static inline void *backend_init( MemoryArena * arena, String8 model_path_arg, s32 *batch_size_restriction, s32 *sr_input_index, s32 *output_dims, s32 *lstm_hidden_size)
{
   return ort_init(arena, model_path_arg, batch_size_restriction, sr_input_index, output_dims, lstm_hidden_size);
}

s32 ort_get_batch_size_restriction( OrtSession * session, OrtAllocator * ort_allocator );
s32 ort_get_output_dims( OrtSession *session, OrtAllocator *ort_allocator );
s32 ort_sr_input_index( OrtSession *session, OrtAllocator *ort_allocator );
s32 ort_lstm_hidden_size( OrtSession *session, OrtAllocator *ort_allocator );
void ort_create_tensors(Silero_Config config, ONNX_Specific *onnx, Tensor_Buffers buffers);
void ort_run(ONNX_Specific *onnx);

static inline void backend_run(MemoryArena *arena, VADC_Context *context)
{
   VAR_UNUSED(arena);
   ort_run(context->backend);
}

static inline void backend_create_tensors(Silero_Config config, void *backend, Tensor_Buffers buffers)
{
   ort_create_tensors(config, backend, buffers);
}
