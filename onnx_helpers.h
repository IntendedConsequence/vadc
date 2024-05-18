#pragma once
#include "include/onnxruntime_c_api.h"


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

static OrtSession *ort_init( MemoryArena * arena, String8 model_path_arg );

s32 ort_get_batch_size_restriction( OrtSession * session, OrtAllocator * ort_allocator );