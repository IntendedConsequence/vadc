#include "vadc.h"


static const OrtApi *g_ort = NULL;

void verify_input_output_count(OrtSession* session) {
   OrtAllocator *allocator;
   ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

   size_t count;
   ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
   for (size_t i = 0; i < count; ++i)
   {
      char *input_name;
      g_ort->SessionGetInputName(session, i, allocator, &input_name);
      printf("Input index %zu name: %s\n", i, input_name);
   }
   // assert(count == 1);

   ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
   for (size_t i = 0; i < count; ++i)
   {
      char *output_name;
      g_ort->SessionGetOutputName(session, i, allocator, &output_name);
      printf("Output index %zu name: %s\n", i, output_name);
   }
   // assert(count == 1);
}

void create_tensor(OrtMemoryInfo* memory_info, OrtValue** out_tensor, int64_t *shape, size_t shape_count, float* input, size_t input_count)
{
   const size_t input_size = input_count * sizeof(float);

   *out_tensor = NULL;
   ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input, input_size, shape,
                                                           shape_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           out_tensor));
   assert(*out_tensor != NULL);
   int is_tensor;
   ORT_ABORT_ON_ERROR(g_ort->IsTensor(*out_tensor, &is_tensor));
   assert(is_tensor);
}

void create_tensor_int64(OrtMemoryInfo* memory_info, OrtValue** out_tensor, int64_t *shape, size_t shape_count, int64_t* input, size_t input_count)
{
   const size_t input_size = input_count * sizeof(int64_t);

   *out_tensor = NULL;
   ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input, input_size, shape,
                                                           shape_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                                           out_tensor));
   assert(*out_tensor != NULL);
   int is_tensor;
   ORT_ABORT_ON_ERROR(g_ort->IsTensor(*out_tensor, &is_tensor));
   assert(is_tensor);
}

int enable_cuda(OrtSessionOptions* session_options)
{
   // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
   OrtCUDAProviderOptions o;
   // Here we use memset to initialize every field of the above data struct to zero.
   memset(&o, 0, sizeof(o));
   // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
   // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
   o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
   o.gpu_mem_limit = SIZE_MAX;
   OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
   if (onnx_status != NULL)
   {
      const char* msg = g_ort->GetErrorMessage(onnx_status);
      fprintf(stderr, "%s\n", msg);
      g_ort->ReleaseStatus(onnx_status);
      return -1;
   }
   return 0;
}

