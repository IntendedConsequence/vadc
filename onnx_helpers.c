#include "onnx_helpers.h"


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

static const wchar_t model_filename[] = SILERO_FILENAME;

static OrtSession *ort_init( MemoryArena *arena, String8 model_path_arg )
{
   g_ort = OrtGetApiBase()->GetApi( ORT_API_VERSION );
   if ( !g_ort )
   {
      fprintf( stderr, "Failed to init ONNX Runtime engine.\n" );
      return 0;
   }

   OrtEnv *env;
   ORT_ABORT_ON_ERROR( g_ort->CreateEnv( ORT_LOGGING_LEVEL_ERROR, "test", &env ) );
   Assert( env != NULL );

   OrtSessionOptions *session_options;
   ORT_ABORT_ON_ERROR( g_ort->CreateSessionOptions( &session_options ) );
   // enable_cuda(session_options);
   ORT_ABORT_ON_ERROR( g_ort->SetIntraOpNumThreads( session_options, 4 ) );
   ORT_ABORT_ON_ERROR( g_ort->SetInterOpNumThreads( session_options, 1 ) );

#define MODEL_PATH_BUFFER_SIZE 1024
   wchar_t *model_path_arg_w = 0;
   String8_ToWidechar( arena, &model_path_arg_w, model_path_arg );

   const size_t model_path_buffer_size = MODEL_PATH_BUFFER_SIZE;
   wchar_t model_path[MODEL_PATH_BUFFER_SIZE];
   GetModuleFileNameW( NULL, model_path, (DWORD)model_path_buffer_size );
   PathRemoveFileSpecW( model_path );
   PathAppendW( model_path, model_path_arg_w ? model_path_arg_w : model_filename );

   OrtSession *session;
   ORT_ABORT_ON_ERROR( g_ort->CreateSession( env, model_path, session_options, &session ) );

   return session;
}

s32 ort_get_batch_size_restriction( OrtSession *session, OrtAllocator *ort_allocator )
{
   s32 batch_size_restriction = 1;

   size_t model_input_count = 0;
   ORT_ABORT_ON_ERROR( g_ort->SessionGetInputCount( session, &model_input_count ) );

   for ( size_t i = 0; i < model_input_count; i++ )
   {
      char *input_name;
      g_ort->SessionGetInputName( session, i, ort_allocator, &input_name );

      if ( strcmp( input_name, "input" ) == 0 )
      {
         OrtTypeInfo *type_info;
         g_ort->SessionGetInputTypeInfo( session, i, &type_info );

         const OrtTensorTypeAndShapeInfo *tensor_info;
         g_ort->CastTypeInfoToTensorInfo( type_info, &tensor_info );

         int64_t dim_value;
         g_ort->GetDimensions( tensor_info, &dim_value, 1 );

         batch_size_restriction = (int)dim_value;

         // fprintf(stderr, "Axis-0 dimension of 'input': %" PRId64 "\n", dim_value);

         g_ort->ReleaseTypeInfo( type_info );
         break;
      }

      g_ort->AllocatorFree( ort_allocator, input_name );
   }

   return batch_size_restriction;
}
