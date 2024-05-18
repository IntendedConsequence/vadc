#pragma once
#include "vadc.h"


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
