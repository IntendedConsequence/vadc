#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "tensor.h"


#include "decoder.c"
#include "lstm.c"

#define MATHS_IMPLEMENTATION
#include "maths.h"

#define MEMORY_IMPLEMENTATION
#include "memory.h"

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

   LoadTesttensorResult result = { 0 };

   // Assert(tensor);
   // memset(tensor, 0, sizeof(*tensor));

   FILE *f = fopen( path, "rb" );
   AssertMessage( f, "Couldn't open file" );

   TestTensor_Header header = { 0 };
   Assert( fread( &header, sizeof( header ), 1, f ) );
   Assert( header.version == 1 );

   int tensor_count = header.tensor_count;
   Assert( tensor_count > 0 );

   TestTensor *tensor_array = pushArray( debug_arena, tensor_count, TestTensor );

   for ( int i = 0; i < tensor_count; ++i )
   {
      TestTensor *tensor = tensor_array + i;
      int name_len = 0;
      Assert( fread( &name_len, sizeof( name_len ), 1, f ) );
      Assert( name_len );
      char *name = pushSizeZeroed( debug_arena, name_len + 1, 1 );
      Assert( fread( name, sizeof( char ), name_len, f ) );
      tensor->name = name;
   }

   for ( int i = 0; i < tensor_count; ++i )
   {
      TestTensor *tensor = tensor_array + i;

      Assert( fread( &tensor->ndim, sizeof( tensor->ndim ), 1, f ) );
      if ( tensor->ndim )
      {
         tensor->dims = pushArray( debug_arena, tensor->ndim, int );
         Assert( fread( tensor->dims, sizeof( tensor->dims[0] ), tensor->ndim, f ) );
      }
      Assert( fread( &tensor->size, sizeof( tensor->size ), 1, f ) );
      Assert( fread( &tensor->nbytes, sizeof( tensor->nbytes ), 1, f ) );

      tensor->data = pushSizeZeroed( debug_arena, tensor->nbytes, 1 );
      Assert( fread( tensor->data, tensor->nbytes, 1, f ) );
   }

   fclose( f );

   result.tensor_array = tensor_array;
   result.tensor_count = tensor_count;

   Assert( result.tensor_array );
   Assert( result.tensor_count );

   return result;
}


static void print_tensors( LoadTesttensorResult res )
{
   fprintf( stderr, "Tensor count: %d\n", res.tensor_count );
   for ( int i = 0; i < res.tensor_count; ++i )
   {
      TestTensor *t = res.tensor_array + i;
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
      int cutoff = t->size <= 10 ? t->size : 10;
      for ( int j = 0; j < cutoff; ++j )
      {
         fprintf( stderr, "%f\n", t->data[j] );
      }

      fprintf( stderr, "\n" );
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
   TestResult result = { 0 };
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

   LoadTesttensorResult res = { 0 };
   res = load_testtensor( "decoder_test.testtensor" );
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

   float atol = 1e-10f;

   TestResult test_result = all_close( result->data, output, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult lstm_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_res = { 0 };
   lstm_res = load_testtensor( "lstm_nito_reference_randn.testtensor" );

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

   lstm_seq(input_x->data,
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

   LoadTesttensorResult lstm_input = { 0 };
   LoadTesttensorResult lstm_weights = { 0 };
   LoadTesttensorResult lstm_output = { 0 };
   lstm_input = load_testtensor( "RED600_all_before_lstm.testtensor" );
   lstm_weights = load_testtensor( "lstm_silero_3.1_16k_for_c.testtensor" );
   lstm_output = load_testtensor( "RED600_all_lstm_output_lite.testtensor" );

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

   lstm_seq(input->data,
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

// TODO(irwin):
// - [ ] move to tensor source files
// - [ ] use where applicable
TestTensor *tensor_zeros_like(MemoryArena *arena, TestTensor *reference)
{
   TestTensor *result = pushStruct(arena, TestTensor);
   result->ndim = reference->ndim;

   static_assert(sizeof(result->dims[0]) == sizeof(int), "ERROR");
   result->dims = pushArray(arena, result->ndim, int);
   for (int i = 0; i < result->ndim; ++i)
   {
      result->dims[i] = reference->dims[i];
   }
   result->nbytes = reference->nbytes;
   result->size = reference->size;
   result->data = pushArray(arena, result->size, float);

   return result;
}

TestResult dw_conv_129_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = { 0 };
   res = load_testtensor( "dw_conv_129.testtensor" );
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *result = res.tensor_array + 3;

   size_t output_size = input->dims[0] * input->dims[1] * sizeof( float );
   Assert( output_size == result->nbytes );

   TestTensor *output_tensor = tensor_zeros_like(debug_arena, result);

   // TODO(irwin): dehardcode 129, put assert instead
   dw_conv_tensor(*input, 129, *weights, *biases, *output_tensor);

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output_tensor->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult pw_conv_129_16_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = { 0 };
   res = load_testtensor( "pw_conv_129_16.testtensor" );
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *result = res.tensor_array + 3;

   size_t output_size = weights->dims[0] * input->dims[1] * sizeof( float );
   Assert( output_size == result->nbytes );

   TestTensor *output_tensor = tensor_zeros_like(debug_arena, result);

   pw_conv_tensor(*input, *weights, *biases, *output_tensor);

   float atol = 1e-4f;

   TestResult test_result = all_close( result->data, output_tensor->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

static const char *result_strings[] =
{
   "FAIL",
   "PASS"
};

typedef TestResult (*TestFunction)();

typedef struct TestFunctionDescription TestFunctionDescription;

struct TestFunctionDescription
{
   TestFunction function_pointer;
   const char *test_name;
   // TODO(irwin): this seems to duplicate the hardcoded atol parameter in each test function's call to all_close
   float acceptable_error_magnitude;
};

TestFunctionDescription test_function_descriptions[] =
{
   { dw_conv_129_test, "dw_conv_129_test", 1e-04f },
   { pw_conv_129_16_test, "pw_conv_129_16_test", 1e-04f },
   { decoder_test, "Decoder", 1e-10f },
   { lstm_test, "LSTM", 1e-04f },
   { lstm_test_RED, "LSTM RED", 1e-04f },
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

      fprintf( stderr, "%s max error magnitude: %s", desc->test_name, test_error_magnitude_names[result.error_magnitude] );
      fprintf( stderr, " ... %s\n", result_strings[!!result.pass] );
   }

   fprintf( stderr, "\n---\n" );
   if ( all_pass )
   {
      fprintf( stderr, "All %d tests PASSED!\n", test_count );
   } else
   {
      fprintf( stderr, "%d out of %d tests FAILED!\n", failed_count, test_count );
   }


   return 0;
}
