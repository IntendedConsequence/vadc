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

static void *ort_init( MemoryArena *arena, String8 model_path_arg, s32 *batch_size_restriction, b32 *is_silero_v4 )
{
   g_ort = OrtGetApiBase()->GetApi( ORT_API_VERSION );
   if ( !g_ort )
   {
      fprintf( stderr, "Failed to init ONNX Runtime engine.\n" );
      return 0;
   }

   OrtEnv *env;
   ORT_ABORT_ON_ERROR( g_ort->CreateEnv( ORT_LOGGING_LEVEL_ERROR, "test", &env ) );
   // ORT_ABORT_ON_ERROR( g_ort->CreateEnv( ORT_LOGGING_LEVEL_VERBOSE, "test", &env ) );
   Assert( env != NULL );
   ORT_ABORT_ON_ERROR( g_ort->DisableTelemetryEvents( env ) );
   // ORT_ABORT_ON_ERROR( g_ort->EnableTelemetryEvents( env ) );

   OrtSessionOptions *session_options;
   ORT_ABORT_ON_ERROR( g_ort->CreateSessionOptions( &session_options ) );
   // enable_cuda(session_options);
   ORT_ABORT_ON_ERROR( g_ort->SetIntraOpNumThreads( session_options, 1 ) );
   ORT_ABORT_ON_ERROR( g_ort->SetInterOpNumThreads( session_options, 1 ) );

#define MODEL_PATH_BUFFER_SIZE 1024
   wchar_t *model_path_arg_w = 0;
   String8_ToWidechar( arena, &model_path_arg_w, model_path_arg );

   const size_t model_path_buffer_size = MODEL_PATH_BUFFER_SIZE;
   wchar_t model_path[MODEL_PATH_BUFFER_SIZE];
   GetModuleFileNameW( NULL, model_path, (DWORD)model_path_buffer_size );
   PathRemoveFileSpecW( model_path );
   PathAppendW( model_path, model_path_arg_w ? model_path_arg_w : model_filename );

   ONNX_Specific *onnx = pushStruct(arena, ONNX_Specific);

   ORT_ABORT_ON_ERROR( g_ort->CreateSession( env, model_path, session_options, &onnx->session ) );

   if (onnx->session)
   {
      ORT_ABORT_ON_ERROR( g_ort->CreateCpuMemoryInfo( OrtArenaAllocator, OrtMemTypeDefault, &onnx->memory_info ) );
      ORT_ABORT_ON_ERROR( g_ort->CreateAllocator( onnx->session, onnx->memory_info, &onnx->ort_allocator ) );

      // onnx->batch_size_restriction = ort_get_batch_size_restriction(onnx->session, onnx->ort_allocator);
      // *batch_size_restriction = onnx->batch_size_restriction;
      *batch_size_restriction = ort_get_batch_size_restriction(onnx->session, onnx->ort_allocator);

      {
         size_t model_output_count = 0;
         size_t model_input_count = 0;

         ORT_ABORT_ON_ERROR( g_ort->SessionGetOutputCount( onnx->session, &model_output_count ) );
         ORT_ABORT_ON_ERROR( g_ort->SessionGetInputCount( onnx->session, &model_input_count ) );

         Assert( model_input_count == 3 || model_input_count == 4 );

         onnx->outputs_count = model_output_count;
         onnx->inputs_count = model_input_count;
         // onnx->is_silero_v4 = model_input_count == 4;
         // *is_silero_v4 = onnx->is_silero_v4;

         // TODO(irwin): dehardcode batch == 1 restriction for silero v4
         // NOTE(irwin): silero v4 model was not yet reexported with contiguous batching support
         if (model_input_count == 4)
         {
            *is_silero_v4 = true;
            // *batch_size_restriction = 1;
         }
         else
         {
            *is_silero_v4 = false;
         }
      }

   }

   return onnx;
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

void ort_create_tensors(Silero_Config config, ONNX_Specific *onnx, Tensor_Buffers buffers)
{
#if SILERO_V5
   int64_t input_tensor_samples_shape[] = {config.batch_size, config.input_count + SILERO_V5_CONTEXT_SIZE};
#else
   int64_t input_tensor_samples_shape[] = {config.batch_size, config.input_count};
#endif // SILERO_V5
   const size_t input_tensor_samples_shape_count = ArrayCount( input_tensor_samples_shape );
   OrtValue **input_tensors = onnx->input_tensors;

   // NOTE(irwin): samples input
#if SILERO_V5
   create_tensor(onnx->memory_info,
                 &input_tensors[0],
                 input_tensor_samples_shape,
                 input_tensor_samples_shape_count,
                 buffers.input_samples,
                 (config.input_count + SILERO_V5_CONTEXT_SIZE) * config.batch_size);
#else
   create_tensor(onnx->memory_info,
                 &input_tensors[0],
                 input_tensor_samples_shape,
                 input_tensor_samples_shape_count,
                 buffers.input_samples,
                 config.input_count * config.batch_size);
#endif // SILERO_V5

#if SILERO_V5
   int64_t state_shape[] = {1, 1, 128};
#else
   int64_t state_shape[] = {2, 1, 64};
#endif // SILERO_V5
   const size_t state_shape_count = ArrayCount( state_shape );
   OrtValue **state_h_tensor = &input_tensors[1];
   // NOTE(irwin): lstm h
   create_tensor(onnx->memory_info,
                 state_h_tensor,
                 state_shape,
                 state_shape_count,
                 buffers.lstm_h,
                 buffers.lstm_count);

   OrtValue** state_c_tensor = &input_tensors[2];
   // NOTE(irwin): lstm c
   create_tensor(onnx->memory_info,
                 state_c_tensor,
                 state_shape,
                 state_shape_count,
                 buffers.lstm_c,
                 buffers.lstm_count);

   if ( config.is_silero_v4 )
   {
      static int64_t sr = 16000;
      // int64_t sr_shape[] = {1, 1};
      // const size_t sr_shape_count = ArrayCount( sr_shape );
      OrtValue **sr_tensor = &input_tensors[3];
   // NOTE(irwin): sample rate
      create_tensor_int64( onnx->memory_info, sr_tensor, 0, 0, &sr, 1 );
   }

   OrtValue **output_tensors = onnx->output_tensors;
   OrtValue **output_prob_tensor = &output_tensors[0];

   // NOTE(irwin): output tensor
   create_tensor(onnx->memory_info,
                 output_prob_tensor,
                 config.prob_shape,
                 config.prob_shape_count,
                 buffers.output,
                 config.prob_tensor_element_count);

   // NOTE(irwin): lstm h output tensor
   OrtValue** state_h_out_tensor = &output_tensors[1];
   create_tensor(onnx->memory_info,
                 state_h_out_tensor,
                 state_shape,
                 state_shape_count,
                 buffers.lstm_h_out,
                 buffers.lstm_count);

   // NOTE(irwin): lstm c output tensor
   OrtValue** state_c_out_tensor = &output_tensors[2];
   create_tensor(onnx->memory_info,
                 state_c_out_tensor,
                 state_shape,
                 state_shape_count,
                 buffers.lstm_c_out,
                 buffers.lstm_count);

   const size_t silero_input_tensor_count = config.is_silero_v4 ? 4 : 3;

   const char **input_names = config.is_silero_v4 ? INPUT_NAMES_V4 : INPUT_NAMES_V3;

   onnx->input_names = input_names;
   onnx->output_names = OUTPUT_NAMES_NORMAL;

   onnx->inputs_count = silero_input_tensor_count;

   // g_ort->ReleaseMemoryInfo(onnx.memory_info);
}

void ort_run(ONNX_Specific *onnx)
{
   ORT_ABORT_ON_ERROR( g_ort->Run( onnx->session,
                                   NULL,
                                   onnx->input_names,
                                   onnx->input_tensors,
                                   onnx->inputs_count,
                                   onnx->output_names,
                                   onnx->outputs_count,
                                   onnx->output_tensors )
   );

   Assert( onnx->output_tensors[0] != NULL );

   int is_tensor;
   ORT_ABORT_ON_ERROR( g_ort->IsTensor( onnx->output_tensors[0], &is_tensor ) );
   Assert( is_tensor );
}
