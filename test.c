#include <stdio.h>
#include <stdlib.h>

#include <tracy\TracyC.h>

#include "utils.h"
#include "tensor.h"


#include "decoder.c"
#include "first_layer.c"
#include "transformer.c"
#include "lstm.c"

#define MATHS_IMPLEMENTATION
#include "maths.h"

#define MEMORY_IMPLEMENTATION
#include "memory.h"

#define STBIW_ASSERT(x) Assert(x)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct TestTensor_Header TestTensor_Header;
struct TestTensor_Header
{
   int version;
   int tensor_count;
};


typedef struct LoadTesttensorResult LoadTesttensorResult;
struct LoadTesttensorResult
{
   int tensor_count;
   TestTensor *tensor_array;
};

LoadTesttensorResult load_testtensor( const char *path )
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();

   LoadTesttensorResult result = {0};

   // Assert(tensor);
   // memset(tensor, 0, sizeof(*tensor));

   FILE *f = fopen( path, "rb" );
   if (!f)
   {
      return result;
   }
   // AssertMessage( f, "Couldn't open file" );

   TestTensor_Header header = {0};
   size_t fread_result = fread( &header, sizeof( header ), 1, f );
   Assert( fread_result );
   Assert( header.version == 1 );

   int tensor_count = header.tensor_count;
   Assert( tensor_count > 0 );

   TestTensor *tensor_array = pushArray( debug_arena, tensor_count, TestTensor );

   for ( int i = 0; i < tensor_count; ++i )
   {
      TestTensor *tensor = tensor_array + i;
      int name_len = 0;
      fread_result = fread( &name_len, sizeof( name_len ), 1, f );
      Assert( fread_result );
      Assert( name_len );
      char *name = pushSizeZeroed( debug_arena, name_len + 1, 1 );
      fread_result = fread( name, sizeof( char ), name_len, f );
      Assert( fread_result );
      tensor->name = name;
   }

   for ( int i = 0; i < tensor_count; ++i )
   {
      TestTensor *tensor = tensor_array + i;

      fread_result = fread( &tensor->ndim, sizeof( tensor->ndim ), 1, f );
      Assert( fread_result );
      if ( tensor->ndim )
      {
         //tensor->dims = pushArray( debug_arena, tensor->ndim, int );
         fread_result = fread( tensor->dims, sizeof( tensor->dims[0] ), tensor->ndim, f );
         Assert( fread_result );
      }
      fread_result = fread( &tensor->size, sizeof( tensor->size ), 1, f );
      Assert( fread_result );
      fread_result = fread( &tensor->nbytes, sizeof( tensor->nbytes ), 1, f );
      Assert( fread_result );

      tensor->data = pushSizeZeroed( debug_arena, tensor->nbytes, 1 );
      fread_result = fread( tensor->data, tensor->nbytes, 1, f );
      Assert( fread_result );
   }

   fclose( f );

   result.tensor_array = tensor_array;
   result.tensor_count = tensor_count;

   Assert( result.tensor_array );
   Assert( result.tensor_count );

   return result;
}

static void print_single_tensor( TestTensor *t )
{
   fprintf( stderr, "%s:\n", t->name );

   fprintf( stderr, "ndim %d\n", t->ndim );
   if ( t->ndim )
   {
      fprintf( stderr, "dims" );
      for ( int ndim = 0; ndim < t->ndim; ++ndim )
      {
         fprintf( stderr, " %d", t->dims[ndim] );
      }
      fprintf( stderr, "\n" );
   }
   fprintf( stderr, "size %d\n", t->size );
   fprintf( stderr, "nbytes %d\n", t->nbytes );
   const int max_print = 10;
   int cutoff = t->size <= max_print ? t->size : max_print;
   for ( int j = 0; j < cutoff; ++j )
   {
      fprintf( stderr, "%f\n", t->data[j] );
   }

   fprintf( stderr, "\n" );
}

static void print_tensors( LoadTesttensorResult res )
{
   fprintf( stderr, "Tensor count: %d\n", res.tensor_count );
   for ( int i = 0; i < res.tensor_count; ++i )
   {
      TestTensor *t = res.tensor_array + i;
      print_single_tensor( t );
   }
}

typedef enum TestErrorMagnitude
{
   TestErrorMagnitude_Zero,
   TestErrorMagnitude_1E_minus10,
   TestErrorMagnitude_1E_minus9,
   TestErrorMagnitude_1E_minus8,
   TestErrorMagnitude_1E_minus7,
   TestErrorMagnitude_1E_minus6,
   TestErrorMagnitude_1E_minus5,
   TestErrorMagnitude_1E_minus4,
   TestErrorMagnitude_1E_minus3,
   TestErrorMagnitude_1E_minus2,
   TestErrorMagnitude_1E_minus1,
   TestErrorMagnitude_1,
   TestErrorMagnitude_Above_1,

   TestErrorMagnitude_COUNT
} TestErrorMagnitude;

typedef struct TestResult TestResult;
struct TestResult
{
   b32 pass;
   float atol;
   float max_error;
   int error_magnitude;
};

static float test_error_magnitudes[TestErrorMagnitude_COUNT] =
{
   0.0f,
   1e-10f,
   1e-9f,
   1e-8f,
   1e-7f,
   1e-6f,
   1e-5f,
   1e-4f,
   1e-3f,
   1e-2f,
   1e-1f,
   1.0f,
   1e1f
};

static const char *test_error_magnitude_names[TestErrorMagnitude_COUNT] =
{
   "zero",
   "1e-10",
   "1e-9",
   "1e-8",
   "1e-7",
   "1e-6",
   "1e-5",
   "1e-4",
   "1e-3",
   "1e-2",
   "1e-1",
   "1",
   "above 1"
};


int get_test_error_magnitude( float value )
{
   for ( int i = 0; i < TestErrorMagnitude_Above_1; ++i )
   {
      if ( value <= test_error_magnitudes[i] )
      {
         return i;
      }
   }

   return TestErrorMagnitude_Above_1;
}



static TestResult all_close( float *left, float *right, int count, float atol )
{
   TestResult result = {0};
   result.atol = atol;

   float max_error = 0.0f;
   for ( int i = 0; i < count; ++i )
   {
      float adiff = fabsf( left[i] - right[i] );
      if ( adiff > max_error )
      {
         max_error = adiff;
      }
   }

   result.max_error = max_error;
   result.pass = max_error < atol;
   result.error_magnitude = get_test_error_magnitude( max_error );

   return result;
}

TestResult decoder_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\decoder_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *result = res.tensor_array + 3;

   size_t output_size = input->dims[0] * weights->dims[0] * sizeof( float );
   Assert( output_size == result->nbytes );

   float *output = pushSizeZeroed( debug_arena, output_size, 1 );
   int output_ndims = result->ndim;
   int *output_dims = pushArray( debug_arena, result->ndim, int );

   int result_ok = decoder( input->data, input->dims, input->ndim, weights->data, weights->dims, weights->ndim, biases->data, biases->dims, biases->ndim, output, output_dims, output_ndims );
   Assert( result_ok );
   VAR_UNUSED( result_ok );

   float atol = 1e-10f;

   TestResult test_result = all_close( result->data, output, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult lstm_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_res = {0};
   lstm_res = load_testtensor( "testdata\\lstm_nito_reference_randn.testtensor" );
   if (lstm_res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   // Assert(memcmp(output, result->data, output_size) == 0);

   // print_tensors(lstm_res);

   TestTensor *input_x = lstm_res.tensor_array + 0;
   TestTensor *input_h = lstm_res.tensor_array + 1;
   TestTensor *input_c = lstm_res.tensor_array + 2;
   TestTensor *weights_transposed = lstm_res.tensor_array + 3;
   TestTensor *lstm_biases = lstm_res.tensor_array + 4;
   TestTensor *output_combined_reference = lstm_res.tensor_array + 5;

   Assert( input_x->ndim == 2 );
   int seq_length = input_x->dims[0];
   int input_size = input_x->dims[1];

   int lstm_output_size = input_size * 4 + input_size * seq_length;

   float *output_combined = pushArray( debug_arena, lstm_output_size, float );

   lstm_seq( input_x->data,
             seq_length,
             input_size,
             input_h->data,
             input_c->data,
             weights_transposed->data,
             lstm_biases->data,
             output_combined
   );

   TestResult pass_lstm = all_close( output_combined_reference->data, output_combined, lstm_output_size, 1e-04f );

   endTemporaryMemory( mark );

   return pass_lstm;
}

TestResult lstm_test_RED()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_input = {0};
   LoadTesttensorResult lstm_weights = {0};
   LoadTesttensorResult lstm_output = {0};

   lstm_input = load_testtensor( "testdata\\untracked\\RED600_all_before_lstm.testtensor" );
   lstm_weights = load_testtensor( "testdata\\untracked\\lstm_silero_3.1_16k_for_c.testtensor" );
   lstm_output = load_testtensor( "testdata\\untracked\\RED600_all_lstm_output_lite.testtensor" );

   if (lstm_input.tensor_count == 0 ||
       lstm_weights.tensor_count == 0 ||
       lstm_output.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }


   TestTensor *input = lstm_input.tensor_array + 0;

   TestTensor *weights = lstm_weights.tensor_array + 0;
   TestTensor *biases = lstm_weights.tensor_array + 1;

   TestTensor *output_reference = lstm_output.tensor_array + 0;

   /*
   - [x] switch to bigger debug default arena since 16Mb won't be enough for 3x11Mb (lstm in, ref out, and our out)
   - [x] alloc input_h, input_c
   - [x] fill it with zeros
   - [x] alloc output_combined for all iterations
   - [ ] alloc output_combined_one for one iteration (or offset into output_combined)
   */
   Assert( input->ndim == 3 );
   int batches = input->dims[0];
   int seq_length = input->dims[1];
   int input_size = input->dims[2];
   int layer_count = weights->dims[0];

   int batch_stride = seq_length * input_size;
   int lstm_output_size = batch_stride * batches + (input_size * layer_count * 2);
   // Assert(lstm_output_size == output_reference->size);

   int hc_size = input_size * layer_count;
   float *input_h_array = pushArray( debug_arena, hc_size, float );
   float *input_c_array = pushArray( debug_arena, hc_size, float );
   float *output_single_batch = pushArray( debug_arena, lstm_output_size, float );
   // float *output_combined = pushArray( debug_arena, batches * seq_length * input_size, float );

   lstm_seq( input->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             weights->data,
             biases->data,
             output_single_batch
   );
   // print_array(output_single_batch, lstm_output_size);

   TestResult pass_lstm = all_close( output_reference->data, output_single_batch, batch_stride * batches, 1e-04f );

   endTemporaryMemory( mark );

   return pass_lstm;
}


TestResult dw_conv_129_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\dw_conv_129.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *result = res.tensor_array + 3;

   size_t output_size = input->dims[0] * input->dims[1] * sizeof( float );
   Assert( output_size == result->nbytes );
   VAR_UNUSED( output_size );

   TestTensor *output_tensor = tensor_zeros_like( debug_arena, result );

   // TODO(irwin): dehardcode 129, put assert instead
   dw_conv_tensor( input, weights, biases, output_tensor );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output_tensor->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult pw_conv_129_16_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\pw_conv_129_16.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *result = res.tensor_array + 3;

   size_t output_size = weights->dims[0] * input->dims[1] * sizeof( float );
   Assert( output_size == result->nbytes );
   VAR_UNUSED( output_size );

   TestTensor *output_tensor = tensor_zeros_like( debug_arena, result );

   pw_conv_tensor( input, weights, biases, output_tensor );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output_tensor->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult first_layer_conv_block_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\first_layer_conv_block.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int test_data_index = 0;
   TestTensor *dw_conv_weights = res.tensor_array + test_data_index++;
   TestTensor *dw_conv_biases = res.tensor_array + test_data_index++;
   TestTensor *pw_conv_weights = res.tensor_array + test_data_index++;
   TestTensor *pw_conv_biases = res.tensor_array + test_data_index++;
   TestTensor *proj_weights = res.tensor_array + test_data_index++;
   TestTensor *proj_biases = res.tensor_array + test_data_index++;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output_tensor = tensor_zeros_like( debug_arena, result );

   conv_block( input, 1,
               dw_conv_weights, dw_conv_biases,
               pw_conv_weights, pw_conv_biases,
               proj_weights, proj_biases,
               output_tensor );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output_tensor->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}


TestResult transpose2d_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   // [1, 2, 3]
   // [4, 5, 6]
   TestTensor *source = tensor_zeros_2d( debug_arena, 2, 3 );
   source->data[0] = 1.0f;
   source->data[1] = 2.0f;
   source->data[2] = 3.0f;
   source->data[3] = 4.0f;
   source->data[4] = 5.0f;
   source->data[5] = 6.0f;

   // [1, 4]
   // [2, 5]
   // [3, 6]
   TestTensor *reference = tensor_zeros_2d( debug_arena, 3, 2 );
   reference->data[0] = 1.0f;
   reference->data[1] = 4.0f;
   reference->data[2] = 2.0f;
   reference->data[3] = 5.0f;
   reference->data[4] = 3.0f;
   reference->data[5] = 6.0f;

   TestTensor *output_tensor = tensor_transpose_last_2d( debug_arena, source );

   float atol = 1e-4f;

   TestResult test_result = all_close( reference->data, output_tensor->data, reference->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}


TestResult softmax_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\softmax_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int test_data_index = 0;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *input_copy = tensor_copy( debug_arena, input );

   softmax_inplace_stable( debug_arena, input_copy );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, input_copy->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult layer_norm_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\layernorm_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   // TODO(irwin): validate loaded tensor count helpers
   Assert( res.tensor_count == 4 );

   int test_data_index = 0;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *weight = res.tensor_array + test_data_index++;
   TestTensor *bias = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   layer_norm( input, weight, bias, output );

   float atol = 1e-4f;
   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult batch_norm_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\batchnorm_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   // TODO(irwin): validate loaded tensor count helpers
   Assert( res.tensor_count == 6 );

   int test_data_index = 0;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *running_mean = res.tensor_array + test_data_index++;
   TestTensor *running_var = res.tensor_array + test_data_index++;
   TestTensor *weight = res.tensor_array + test_data_index++;
   TestTensor *bias = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   batch_norm1d( input, running_mean, running_var, weight, bias, output );

   float atol = 1e-4f;
   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult stft_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\untracked\\stft_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   // TODO(irwin): validate loaded tensor count helpers
   Assert( res.tensor_count == 3 );

   int test_data_index = 0;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *forward_basis_buffer = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   my_stft(debug_arena, input, forward_basis_buffer, output );

   float atol = 1e-4f;
   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult adaptive_audio_normalization_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\adaptive_audio_normalization_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   // TODO(irwin): validate loaded tensor count helpers
   Assert( res.tensor_count == 2 );

   int test_data_index = 0;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_copy( debug_arena, input );

   adaptive_audio_normalization_inplace(debug_arena, output );

   float atol = 1e-4f;
   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}


TestResult dual_head_attention_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\dual_head_attention_test.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int test_data_index = 0;
   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *weights = res.tensor_array + test_data_index++;
   TestTensor *biases = res.tensor_array + test_data_index++;
   TestTensor *proj_weights = res.tensor_array + test_data_index++;
   TestTensor *proj_biases = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output_tensor = tensor_zeros_like( debug_arena, result );

   dual_head_attention( input,
                        weights, biases,
                        proj_weights, proj_biases,
                        output_tensor );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output_tensor->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult transformer_block_16_16_48_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\transformer_block_test_16_16_48.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int test_data_index = 0;
   TestTensor *attention_weights = res.tensor_array + test_data_index++;
   TestTensor *attention_biases = res.tensor_array + test_data_index++;
   TestTensor *attention_proj_weights = res.tensor_array + test_data_index++;
   TestTensor *attention_proj_biases = res.tensor_array + test_data_index++;
   TestTensor *norm1_weights = res.tensor_array + test_data_index++;
   TestTensor *norm1_biases = res.tensor_array + test_data_index++;
   TestTensor *norm2_weights = res.tensor_array + test_data_index++;
   TestTensor *norm2_biases = res.tensor_array + test_data_index++;
   TestTensor *linear1_weights = res.tensor_array + test_data_index++;
   TestTensor *linear1_biases = res.tensor_array + test_data_index++;
   TestTensor *linear2_weights = res.tensor_array + test_data_index++;
   TestTensor *linear2_biases = res.tensor_array + test_data_index++;

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   // NOTE(irwin): 16, 16, 48
   transformer_block( debug_arena, input,
                      attention_weights, attention_biases,
                      attention_proj_weights, attention_proj_biases,
                      norm1_weights, norm1_biases,
                      linear1_weights, linear1_biases,
                      linear2_weights, linear2_biases,
                      norm2_weights, norm2_biases,
                      output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

static inline int fill_transformer_weights( TransformerLayer_Weights *weights, TestTensor *tensor_array, b32 has_out_proj )
{
   int test_data_index = 0;

   weights->dw_conv_weights = tensor_array + test_data_index++;
   weights->dw_conv_biases = tensor_array + test_data_index++;
   weights->pw_conv_weights = tensor_array + test_data_index++;
   weights->pw_conv_biases = tensor_array + test_data_index++;

   if ( has_out_proj )
   {
      weights->proj_weights = tensor_array + test_data_index++;
      weights->proj_biases = tensor_array + test_data_index++;
   }

   weights->attention_weights = tensor_array + test_data_index++;
   weights->attention_biases = tensor_array + test_data_index++;
   weights->attention_proj_weights = tensor_array + test_data_index++;
   weights->attention_proj_biases = tensor_array + test_data_index++;

   weights->norm1_weights = tensor_array + test_data_index++;
   weights->norm1_biases = tensor_array + test_data_index++;
   weights->linear1_weights = tensor_array + test_data_index++;
   weights->linear1_biases = tensor_array + test_data_index++;
   weights->linear2_weights = tensor_array + test_data_index++;
   weights->linear2_biases = tensor_array + test_data_index++;
   weights->norm2_weights = tensor_array + test_data_index++;
   weights->norm2_biases = tensor_array + test_data_index++;

   weights->conv_weights = tensor_array + test_data_index++;
   weights->conv_biases = tensor_array + test_data_index++;

   weights->batch_norm_weights = tensor_array + test_data_index++;
   weights->batch_norm_biases = tensor_array + test_data_index++;
   weights->batch_norm_running_mean = tensor_array + test_data_index++;
   weights->batch_norm_running_var = tensor_array + test_data_index++;

   return test_data_index;
}

static inline int fill_encoder_weights(Encoder_Weights *encoder_weights, TestTensor *tensor_array)
{
   int test_data_index = 0;

   encoder_weights->l1_conv_stride = 2;
   encoder_weights->l2_conv_stride = 2;
   encoder_weights->l3_conv_stride = 1;
   encoder_weights->l4_conv_stride = 1;


   test_data_index += fill_transformer_weights( &encoder_weights->l1, tensor_array + test_data_index, true );
   test_data_index += fill_transformer_weights( &encoder_weights->l2, tensor_array + test_data_index, true );
   test_data_index += fill_transformer_weights( &encoder_weights->l3, tensor_array + test_data_index, false );
   test_data_index += fill_transformer_weights( &encoder_weights->l4, tensor_array + test_data_index, true );

   return test_data_index;
}

TestResult transformer_first_layer_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\transformer_first_layer.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   Assert( res.tensor_count == 26 );

   TransformerLayer_Weights transformer_weights = {0};

   int test_data_index = 0;
   test_data_index += fill_transformer_weights( &transformer_weights, res.tensor_array + test_data_index, true );

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   // NOTE(irwin): 16, 16, 48
   transformer_layer( debug_arena,
                      input,
                      transformer_weights,
                      2,
                      output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}


TestResult transformer_layers_1_2_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\transformer_layers_1_2.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   Assert( res.tensor_count == (24 + 24 + 2) );

   TransformerLayer_Weights transformer_weights_l1 = {0};
   TransformerLayer_Weights transformer_weights_l2 = {0};

   int test_data_index = 0;
   test_data_index += fill_transformer_weights( &transformer_weights_l1, res.tensor_array + test_data_index, true );
   test_data_index += fill_transformer_weights( &transformer_weights_l2, res.tensor_array + test_data_index, true );


   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *l1_output = 0;
   {
      ConvOutputShape conv_block_out_shape = conv_block_output_shape( input, transformer_weights_l1.dw_conv_weights, transformer_weights_l1.pw_conv_weights );
      ConvOutputShape l1_output_required_shape = conv_output_shape_shape( conv_block_out_shape, transformer_weights_l1.conv_weights, 2 );
      l1_output = tensor_zeros_3d( debug_arena, l1_output_required_shape.batch_size, l1_output_required_shape.channels_out, l1_output_required_shape.sequence_length );
   }

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   transformer_layer( debug_arena,
                      input,
                      transformer_weights_l1,
                      2,
                      l1_output );

   transformer_layer( debug_arena,
                      l1_output,
                      transformer_weights_l2,
                      2,
                      output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

static inline void dump_tensor_hdr(const char *filename, TestTensor *tensor)
{
   int w = tdim( tensor, -1 );
   int h = tensor->size / w;

   float ar = (float)mymin( w, h ) / mymax( w, h );
   for (int i = 1; i < tensor->size; ++i)
   {
      if (tensor->size % i == 0)
      {
         int w2 = i;
         int h2 = tensor->size / w2;
         float ar2 = (float)mymin( w2, h2 ) / mymax( w2, h2 );
         if (ar2 > ar)
         {
            ar = ar2;
            w = w2;
            h = h2;
         }
      }
   }
   stbi_write_hdr( filename, w, h, 1, tensor->data );
}

TestResult transformer_layers_1_2_3_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\transformer_layers_1_2_3.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   Assert( res.tensor_count == (24 + 24 + 22 + 2) );

   TransformerLayer_Weights transformer_weights_l1 = {0};
   TransformerLayer_Weights transformer_weights_l2 = {0};
   TransformerLayer_Weights transformer_weights_l3 = {0};

   int test_data_index = 0;
   test_data_index += fill_transformer_weights( &transformer_weights_l1, res.tensor_array + test_data_index, true );
   test_data_index += fill_transformer_weights( &transformer_weights_l2, res.tensor_array + test_data_index, true );
   test_data_index += fill_transformer_weights( &transformer_weights_l3, res.tensor_array + test_data_index, false );


   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *l1_output = 0;
   {
      ConvOutputShape conv_block_out_shape = conv_block_output_shape( input, transformer_weights_l1.dw_conv_weights, transformer_weights_l1.pw_conv_weights );
      ConvOutputShape l1_output_required_shape = conv_output_shape_shape( conv_block_out_shape, transformer_weights_l1.conv_weights, 2 );
      l1_output = tensor_zeros_3d( debug_arena, l1_output_required_shape.batch_size, l1_output_required_shape.channels_out, l1_output_required_shape.sequence_length );
   }

   TestTensor *l2_output = 0;
   {
      ConvOutputShape conv_block_out_shape = conv_block_output_shape( l1_output, transformer_weights_l2.dw_conv_weights, transformer_weights_l2.pw_conv_weights );
      ConvOutputShape l2_output_required_shape = conv_output_shape_shape( conv_block_out_shape, transformer_weights_l2.conv_weights, 2 );
      l2_output = tensor_zeros_3d( debug_arena, l2_output_required_shape.batch_size, l2_output_required_shape.channels_out, l2_output_required_shape.sequence_length );
   }

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   transformer_layer( debug_arena,
                      input,
                      transformer_weights_l1,
                      2,
                      l1_output );

   transformer_layer( debug_arena,
                      l1_output,
                      transformer_weights_l2,
                      2,
                      l2_output );

   transformer_layer( debug_arena,
                      l2_output,
                      transformer_weights_l3,
                      1,
                      output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult transformer_layers_1_2_3_4_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\transformer_layers_1_2_3_4.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == (encoder_weights_count + 2) );

   int test_data_index = 0;

   // NOTE(irwin): encoder weights
   Encoder_Weights encoder_weights = {0};
   int encoder_weights_read = fill_encoder_weights( &encoder_weights, res.tensor_array + test_data_index );
   Assert( encoder_weights_read == encoder_weights_count );
   test_data_index += encoder_weights_read;

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   ConvOutputShape l4_output_required_shape = shape_for_encoder( input, encoder_weights );
   TestTensor *output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

   encoder( debug_arena, input, encoder_weights, output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult adaptive_normalization_encoder_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\adaptive_normalization_encoder.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == (encoder_weights_count + 2) );


   int test_data_index = 0;

   // NOTE(irwin): encoder weights
   Encoder_Weights encoder_weights = {0};
   int encoder_weights_read = fill_encoder_weights( &encoder_weights, res.tensor_array + test_data_index );
   Assert( encoder_weights_read == encoder_weights_count );
   test_data_index += encoder_weights_read;

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *normalization_output = tensor_copy( debug_arena, input );
   adaptive_audio_normalization_inplace( debug_arena, normalization_output );

   ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, encoder_weights );
   TestTensor *output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

   encoder( debug_arena, normalization_output, encoder_weights, output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult stft_normalization_encoder_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\untracked\\stft_normalization_encoder.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == (1 + encoder_weights_count + 2) );


   int test_data_index = 0;
   TestTensor *forward_basis_buffer = res.tensor_array + test_data_index++;

   // NOTE(irwin): encoder weights
   Encoder_Weights encoder_weights = {0};
   int encoder_weights_read = fill_encoder_weights( &encoder_weights, res.tensor_array + test_data_index );
   Assert( encoder_weights_read == encoder_weights_count );
   test_data_index += encoder_weights_read;

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   int cutoff;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      int half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input, forward_basis_buffer, 64 );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input, forward_basis_buffer, stft_output );

   TestTensor *normalization_output = tensor_copy( debug_arena, stft_output );
   adaptive_audio_normalization_inplace( debug_arena, normalization_output );

   ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, encoder_weights );
   TestTensor *output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

   encoder( debug_arena, normalization_output, encoder_weights, output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult stft_normalization_encoder_lstm_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   LoadTesttensorResult lstm_weights_res = {0};

   res = load_testtensor( "testdata\\untracked\\stft_normalization_encoder_lstm.testtensor" );
   lstm_weights_res = load_testtensor( "testdata\\untracked\\lstm_silero_3.1_16k_for_c.testtensor" );

   if ( res.tensor_count == 0 || lstm_weights_res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == (1 + encoder_weights_count + 2) );


   int test_data_index = 0;
   TestTensor *forward_basis_buffer = res.tensor_array + test_data_index++;

   // NOTE(irwin): encoder weights
   Encoder_Weights encoder_weights = {0};
   int encoder_weights_read = fill_encoder_weights( &encoder_weights, res.tensor_array + test_data_index );
   Assert( encoder_weights_read == encoder_weights_count );
   test_data_index += encoder_weights_read;

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *lstm_weights = lstm_weights_res.tensor_array + 0;
   TestTensor *lstm_biases = lstm_weights_res.tensor_array + 1;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   int cutoff;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      int half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input, forward_basis_buffer, 64 );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input, forward_basis_buffer, stft_output );

   TestTensor *normalization_output = tensor_copy( debug_arena, stft_output );
   adaptive_audio_normalization_inplace( debug_arena, normalization_output );

   ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, encoder_weights );
   TestTensor *l4_output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

   encoder( debug_arena, normalization_output, encoder_weights, l4_output );

   TestTensor *l4_output_t = tensor_transpose_last_2d(debug_arena, l4_output);

   int batches = tdim( l4_output_t, -3);
   int seq_length = tdim( l4_output_t, -2);
   int input_size = tdim( l4_output_t, -1);
   int layer_count = tdim(lstm_weights, 0);

   int batch_stride = seq_length * input_size;
   int lstm_output_size = batch_stride * batches + (input_size * layer_count * 2);
   Assert(lstm_output_size == output->size);

   int hc_size = input_size * layer_count;
   float *input_h_array = pushArray( debug_arena, hc_size, float );
   float *input_c_array = pushArray( debug_arena, hc_size, float );
   //float *output_single_batch = pushArray( debug_arena, lstm_output_size, float );
   // float *output_combined = pushArray( debug_arena, batches * seq_length * input_size, float );

   lstm_seq( l4_output_t->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             lstm_weights->data,
             lstm_biases->data,
             output->data
   );


   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult stft_normalization_encoder_lstm_decoder_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   LoadTesttensorResult lstm_weights_res = {0};

   res = load_testtensor( "testdata\\untracked\\stft_normalization_encoder_lstm_decoder.testtensor" );
   lstm_weights_res = load_testtensor( "testdata\\untracked\\lstm_silero_3.1_16k_for_c.testtensor" );

   if ( res.tensor_count == 0 || lstm_weights_res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == (1 + encoder_weights_count + 2 + 2) );


   int test_data_index = 0;
   TestTensor *forward_basis_buffer = res.tensor_array + test_data_index++;

   // NOTE(irwin): encoder weights
   Encoder_Weights encoder_weights = {0};
   int encoder_weights_read = fill_encoder_weights( &encoder_weights, res.tensor_array + test_data_index );
   Assert( encoder_weights_read == encoder_weights_count );
   test_data_index += encoder_weights_read;

   TestTensor *decoder_weights = res.tensor_array + test_data_index++;
   TestTensor *decoder_biases = res.tensor_array + test_data_index++;

   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *lstm_weights = lstm_weights_res.tensor_array + 0;
   TestTensor *lstm_biases = lstm_weights_res.tensor_array + 1;

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   int cutoff;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      int half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input, forward_basis_buffer, 64 );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input, forward_basis_buffer, stft_output );

   TestTensor *normalization_output = tensor_copy( debug_arena, stft_output );
   adaptive_audio_normalization_inplace( debug_arena, normalization_output );

   ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, encoder_weights );
   TestTensor *l4_output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

   encoder( debug_arena, normalization_output, encoder_weights, l4_output );

   TestTensor *l4_output_t = tensor_transpose_last_2d(debug_arena, l4_output);

   int batches = tdim( l4_output_t, -3);
   int seq_length = tdim( l4_output_t, -2);
   int input_size = tdim( l4_output_t, -1);
   int layer_count = tdim(lstm_weights, 0);

   int batch_stride = seq_length * input_size;
   int lstm_output_size = batch_stride * batches + (input_size * layer_count * 2);
   // Assert(lstm_output_size == output->size);

   int hc_size = input_size * layer_count;
   float *input_h_array = pushArray( debug_arena, hc_size, float );
   float *input_c_array = pushArray( debug_arena, hc_size, float );
   float *lstm_output = pushArray( debug_arena, lstm_output_size, float );
   //float *lstm_output = pushArray( debug_arena, batches * seq_length * input_size, float );

   lstm_seq( l4_output_t->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             lstm_weights->data,
             lstm_biases->data,
             lstm_output
   );

   TestTensor *lstm_output_tensor = tensor_zeros_3d( debug_arena, batches, seq_length, input_size );
   memmove( lstm_output_tensor->data, lstm_output, lstm_output_tensor->nbytes );
   TestTensor *lstm_output_tensor_t = tensor_transpose_last_2d( debug_arena, lstm_output_tensor );

   int decoder_output_size = batches * tdim( decoder_weights, 0 );
   Assert( decoder_output_size == output->size );

   int decoder_result = decoder_tensor( lstm_output_tensor_t, decoder_weights, decoder_biases, output );
   VAR_UNUSED( decoder_result );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}


TestResult silero_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   LoadTesttensorResult silero_weights_res = {0};

   res = load_testtensor( "testdata\\untracked\\silero.testtensor" );
   silero_weights_res = load_testtensor( "testdata\\silero_v31_16k.testtensor" );

   if ( res.tensor_count == 0 || silero_weights_res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == (1 + encoder_weights_count + 2 + 2) );
   Assert( silero_weights_res.tensor_count == (1 + encoder_weights_count + 2 + 2) );

   int silero_weights_index = 0;
   TestTensor *forward_basis_buffer = silero_weights_res.tensor_array + silero_weights_index++;

   // NOTE(irwin): encoder weights
   Encoder_Weights encoder_weights = {0};
   int encoder_weights_read = fill_encoder_weights( &encoder_weights, silero_weights_res.tensor_array + silero_weights_index );
   Assert( encoder_weights_read == encoder_weights_count );
   silero_weights_index += encoder_weights_read;

   // NOTE(irwin): lstm weights
   TestTensor *lstm_weights = silero_weights_res.tensor_array + silero_weights_index++;
   TestTensor *lstm_biases = silero_weights_res.tensor_array + silero_weights_index++;

   // NOTE(irwin): decoder weights
   TestTensor *decoder_weights = silero_weights_res.tensor_array + silero_weights_index++;
   TestTensor *decoder_biases = silero_weights_res.tensor_array + silero_weights_index++;

   int test_data_index = 0;
   test_data_index++; // skip forward_basis_buffer
   test_data_index += encoder_weights_count; // skip encoder weights
   test_data_index += 2; // skip decoder weights

   TestTensor *input_batches = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;


   TestTensor *lstm_input_h = tensor_zeros_3d(debug_arena, 2, 1, 64);
   TestTensor *lstm_input_c = tensor_zeros_like(debug_arena, lstm_input_h);

   TestTensor *lstm_output_h = tensor_zeros_like( debug_arena, lstm_input_h );
   TestTensor *lstm_output_c = tensor_zeros_like( debug_arena, lstm_input_h );

   TestTensor *output = tensor_zeros_like( debug_arena, result );

   int batch_count = tdim( input_batches, 0 );
#if 0
   int cutoff;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      int half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input_batches, forward_basis_buffer, 64 );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input_batches, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input_batches, forward_basis_buffer, stft_output );

   for ( int batch_index = 0; batch_index < batch_count; ++batch_index )
   {
      TemporaryMemory batch_mark = beginTemporaryMemory( debug_arena );

      TestTensor input_one_batch = tensor_index_first_dim( stft_output, batch_index, true );


      TestTensor *normalization_output = tensor_copy( debug_arena, &input_one_batch );
#else
   for ( int batch_index = 0; batch_index < batch_count; ++batch_index )
   {
      TemporaryMemory batch_mark = beginTemporaryMemory( debug_arena );

      TestTensor input_one_batch = tensor_index_first_dim( input_batches, batch_index, true );

      int cutoff;
      {
         int filter_length = tdim( forward_basis_buffer, 2 );
         int half_filter_length = filter_length / 2;
         cutoff = half_filter_length + 1;
      }
      // TODO(irwin): dehardcode 64 hop_length
      int stft_out_features_count = compute_stft_output_feature_count( &input_one_batch, forward_basis_buffer, 64 );
      TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( &input_one_batch, -2 ), cutoff, stft_out_features_count );

      my_stft( debug_arena, &input_one_batch, forward_basis_buffer, stft_output );

      TestTensor *normalization_output = tensor_copy( debug_arena, stft_output );
#endif
      adaptive_audio_normalization_inplace( debug_arena, normalization_output );

      ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, encoder_weights );
      TestTensor *l4_output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

      encoder( debug_arena, normalization_output, encoder_weights, l4_output );

      TestTensor *l4_output_t = tensor_transpose_last_2d( debug_arena, l4_output );

      int batches = tdim( l4_output_t, -3 );
      int seq_length = tdim( l4_output_t, -2 );
      int input_size = tdim( l4_output_t, -1 );
      int layer_count = tdim( lstm_weights, 0 );
      int hidden_size = tdim( lstm_weights, -1 ) / 2;
      Assert( hidden_size == input_size );
      Assert( hidden_size == tdim( lstm_biases, -1 ) / 4 );
      int batch_stride = seq_length * input_size;
      int lstm_output_size = batch_stride * batches + (input_size * layer_count * 2);

      int hc_size = input_size * layer_count;
      Assert( hc_size == lstm_input_h->size );

      float *lstm_output = pushArray( debug_arena, lstm_output_size, float );
      //float *lstm_output = pushArray( debug_arena, batches * seq_length * input_size, float );

      lstm_seq( l4_output_t->data,
                seq_length * batches,
                input_size,
                lstm_input_h->data,
                lstm_input_c->data,
                lstm_weights->data,
                lstm_biases->data,
                lstm_output
      );

      // TODO(irwin):
      // lstm output is [7, 64]
      //              + [2, 64] h
      //              + [2, 64] c
      // doesn't support proper batches
      // calls batches what are actually seq_length
      TestTensor *lstm_output_tensor = tensor_zeros_3d( debug_arena, 1, seq_length, input_size );
      memmove( lstm_output_tensor->data, lstm_output, lstm_output_tensor->nbytes );

      memmove( lstm_output_h->data, lstm_output + lstm_output_tensor->size, lstm_output_h->nbytes );
      memmove( lstm_output_c->data, lstm_output + lstm_output_tensor->size + lstm_output_h->size, lstm_output_c->nbytes );

      TestTensor *lstm_output_tensor_t = tensor_transpose_last_2d( debug_arena, lstm_output_tensor );

      int decoder_output_size = batches * tdim( decoder_weights, 0 );

      int decoder_results = tdim( decoder_weights, 0 );
      TestTensor *output_decoder = tensor_zeros_3d( debug_arena, 1, decoder_results, 1 );
      Assert( decoder_output_size == output_decoder->size );

      int decoder_result = decoder_tensor( lstm_output_tensor_t, decoder_weights, decoder_biases, output_decoder );
      VAR_UNUSED( decoder_result );

      float diarization_maybe = output_decoder->data[0];
      float speech_probability = output_decoder->data[1];

      endTemporaryMemory( batch_mark );

      output->data[batch_index * decoder_results + 0] = diarization_maybe;
      output->data[batch_index * decoder_results + 1] = speech_probability;

      memmove( lstm_input_h->data, lstm_output_h->data, lstm_input_h->nbytes );
      memmove( lstm_input_c->data, lstm_output_c->data, lstm_input_c->nbytes );
   }

   float atol = 1e-3f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

#if 0
   if ( test_result.max_error > atol )
   {
      //const char *funcname = __FUNCTION__;
      dump_tensor_hdr( "output.hdr", output );
      dump_tensor_hdr( "output_expected.hdr", result );
   }
#endif


   endTemporaryMemory( mark );

   return test_result;
}

TestResult transformer_layers_3_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor( "testdata\\transformer_layers_3.testtensor" );
   if ( res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   Assert( res.tensor_count == (22 + 2) );

   TransformerLayer_Weights transformer_weights_l3 = {0};

   int test_data_index = 0;
   test_data_index += fill_transformer_weights( &transformer_weights_l3, res.tensor_array + test_data_index, false );


   TestTensor *input = res.tensor_array + test_data_index++;
   TestTensor *result = res.tensor_array + test_data_index++;

   TestTensor *output = tensor_zeros_like( debug_arena, result );


   transformer_layer( debug_arena,
                      input,
                      transformer_weights_l3,
                      1,
                      output );

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output->data, result->size, atol );

#if 0
   if ( test_result.max_error > atol )
   {
      //const char *funcname = __FUNCTION__;
      dump_tensor_hdr( "output.hdr", output );
      dump_tensor_hdr( "output_expected.hdr", result );
   }
#endif

   endTemporaryMemory( mark );

   return test_result;
}

static const char *result_strings[] =
{
   "FAIL",
   "PASS"
};

typedef TestResult ( *TestFunction )();

typedef struct TestFunctionDescription TestFunctionDescription;

struct TestFunctionDescription
{
   TestFunction function_pointer;
   const char *test_name;
};

#define TEST_FUNCTION_DESCRIPTION( test_function ) \
   { test_function, VADC_TOSTRING( test_function ) }

TestFunctionDescription test_function_descriptions[] =
{
   TEST_FUNCTION_DESCRIPTION(dw_conv_129_test),
   TEST_FUNCTION_DESCRIPTION(pw_conv_129_16_test),
   TEST_FUNCTION_DESCRIPTION(first_layer_conv_block_test),
   TEST_FUNCTION_DESCRIPTION(decoder_test),
   TEST_FUNCTION_DESCRIPTION(transpose2d_test),
   TEST_FUNCTION_DESCRIPTION(softmax_test),
   TEST_FUNCTION_DESCRIPTION(layer_norm_test),
   TEST_FUNCTION_DESCRIPTION(batch_norm_test),
   TEST_FUNCTION_DESCRIPTION(dual_head_attention_test),
   TEST_FUNCTION_DESCRIPTION(transformer_block_16_16_48_test),
   TEST_FUNCTION_DESCRIPTION( transformer_first_layer_test ),
   TEST_FUNCTION_DESCRIPTION( transformer_layers_1_2_test ),
   TEST_FUNCTION_DESCRIPTION( transformer_layers_3_test ),
   TEST_FUNCTION_DESCRIPTION( transformer_layers_1_2_3_test ),
   TEST_FUNCTION_DESCRIPTION( transformer_layers_1_2_3_4_test ),
   TEST_FUNCTION_DESCRIPTION( adaptive_normalization_encoder_test ),
   TEST_FUNCTION_DESCRIPTION( stft_normalization_encoder_test ),
   TEST_FUNCTION_DESCRIPTION( stft_normalization_encoder_lstm_test ),
   TEST_FUNCTION_DESCRIPTION( stft_normalization_encoder_lstm_decoder_test ),
   TEST_FUNCTION_DESCRIPTION( silero_test ),
   TEST_FUNCTION_DESCRIPTION(stft_test),
   TEST_FUNCTION_DESCRIPTION(adaptive_audio_normalization_test),
   TEST_FUNCTION_DESCRIPTION(lstm_test),
   TEST_FUNCTION_DESCRIPTION(lstm_test_RED),
};

// int main(int argc, char *argv[])
int main()
{
   b32 all_pass = 1;
   int failed_count = 0;
   int passed_count = 0;

   int test_count = ArrayCount( test_function_descriptions );
   fprintf( stderr, "Total tests to run: %d\n", test_count );

   for ( int i = 0; i < test_count; ++i )
   {
      TestFunctionDescription *desc = test_function_descriptions + i;
      TestResult result = desc->function_pointer();

      all_pass &= result.pass;
      passed_count += !!result.pass;
      failed_count += !result.pass;

      fprintf( stderr, "%-34s max error magnitude: %-6s", desc->test_name, test_error_magnitude_names[result.error_magnitude] );
      fprintf( stderr, " ... %s\n", result_strings[!!result.pass] );
   }

   fprintf( stderr, "\n---\n" );
   if ( all_pass )
   {
      fprintf( stderr, "All %d tests PASSED!\n", test_count );
   }
   else
   {
      fprintf( stderr, "%d out of %d tests FAILED!\n", failed_count, test_count );
   }


   return 0;
}
