#include <stdio.h>
#include <stdlib.h>

#include <tracy\TracyC.h>

#if !defined(VADC_SLOW)
#define VADC_SLOW 0
#endif // VADC_SLOW

#include "utils.h"
#include "tensor.h"


#include "conv.c"
#include "misc.c"
#include "stft.c"
#include "lstm.c"
#include "transformer.c"
#include "silero_v3.c"

#define MATHS_IMPLEMENTATION
#include "maths.h"

#define MEMORY_IMPLEMENTATION
#include "memory.h"

#define STBIW_ASSERT(x) Assert(x)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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

static inline void dump_tensor_hdr(const char *filename, TestTensor *tensor);

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
   res = load_testtensor(debug_arena, "testdata\\decoder_test.testtensor" );
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

   decoder(debug_arena, input->data, input->dims, input->ndim, weights->data, weights->dims, weights->ndim, biases->data, biases->dims, biases->ndim, output, output_dims, output_ndims );


   float atol = 1e-10f;

   TestResult test_result = all_close( result->data, output, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult decoder_test_v5()
{
   MemoryArena *arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( arena );

   LoadTesttensorResult decoder_testdata = {0};

   decoder_testdata = load_testtensor(arena, "testdata\\untracked\\v5_decoder.testtensor" );

   if (decoder_testdata.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }


   TestTensor *input = decoder_testdata.tensor_array + 0;

   TestTensor *weights = decoder_testdata.tensor_array + 1;
   TestTensor *biases = decoder_testdata.tensor_array + 2;

   TestTensor *output_reference = decoder_testdata.tensor_array + 3;

   TestTensor *relu_out = tensor_copy(arena, input);
   tensor_relu_inplace(relu_out);
   TestTensor *conv_out = conv_tensor_out(arena, relu_out, weights, biases, 1);
   mysigmoid_inplace(conv_out->data, conv_out->size);

   TestResult pass_lstm = all_close( output_reference->data, conv_out->data, output_reference->size, 1e-04f );

   endTemporaryMemory( mark );

   return pass_lstm;
}

TestResult lstm_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_res = {0};
   lstm_res = load_testtensor(debug_arena, "testdata\\lstm_nito_reference_randn.testtensor" );
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

   lstm_seq( debug_arena, input_x->data,
             seq_length,
             input_size,
             input_h->data,
             input_c->data,
             weights_transposed->data,
             lstm_biases->data,
             output_combined,
             2
   );

   TestResult pass_lstm = all_close( output_combined_reference->data, output_combined, output_combined_reference->size, 1e-04f );

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

   lstm_input = load_testtensor(debug_arena, "testdata\\untracked\\RED600_all_before_lstm.testtensor" );
   lstm_weights = load_testtensor(debug_arena, "testdata\\untracked\\lstm_silero_3.1_16k_for_c.testtensor" );
   lstm_output = load_testtensor(debug_arena, "testdata\\untracked\\RED600_all_lstm_output_lite.testtensor" );

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

   lstm_seq( debug_arena, input->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             weights->data,
             biases->data,
             output_single_batch,
             layer_count
   );
   // print_array(output_single_batch, lstm_output_size);

   TestResult pass_lstm = all_close( output_reference->data, output_single_batch, output_reference->size, 1e-04f );

   endTemporaryMemory( mark );

   return pass_lstm;
}


// NOTE(irwin): same as old but in one file, and input audio wasn't normalized before stft
TestResult lstm_test_RED_new()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_testdata = {0};

   lstm_testdata = load_testtensor(debug_arena, "testdata\\untracked\\RED600_lstm.testtensor" );

   if (lstm_testdata.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }


   TestTensor *input = lstm_testdata.tensor_array + 0;

   TestTensor *weights = lstm_testdata.tensor_array + 1;
   TestTensor *biases = lstm_testdata.tensor_array + 2;

   TestTensor *output_reference = lstm_testdata.tensor_array + 3;

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

   lstm_seq( debug_arena, input->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             weights->data,
             biases->data,
             output_single_batch,
             layer_count
   );
   // print_array(output_single_batch, lstm_output_size);

   TestResult pass_lstm = all_close( output_reference->data, output_single_batch, batch_stride * batches, 1e-04f );

   endTemporaryMemory( mark );

   return pass_lstm;
}

TestResult lstm_test_RED_1layer()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_testdata = {0};

   lstm_testdata = load_testtensor(debug_arena, "testdata\\untracked\\RED600_lstm_1layer.testtensor" );

   if (lstm_testdata.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }


   TestTensor *input = lstm_testdata.tensor_array + 0;

   TestTensor *weights = lstm_testdata.tensor_array + 1;
   TestTensor *biases = lstm_testdata.tensor_array + 2;

   TestTensor *output_reference = lstm_testdata.tensor_array + 3;

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

   lstm_seq( debug_arena, input->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             weights->data,
             biases->data,
             output_single_batch,
             layer_count
   );
   // print_array(output_single_batch, lstm_output_size);

   TestResult pass_lstm = all_close( output_reference->data, output_single_batch, batch_stride * batches, 1e-04f );

   endTemporaryMemory( mark );

   return pass_lstm;
}

TestResult lstm_test_RED_v5()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult lstm_testdata = {0};

   lstm_testdata = load_testtensor(debug_arena, "testdata\\untracked\\RED600_lstm_v5.testtensor" );

   if (lstm_testdata.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }


   TestTensor *input = lstm_testdata.tensor_array + 0;

   TestTensor *weights = lstm_testdata.tensor_array + 1;
   TestTensor *biases = lstm_testdata.tensor_array + 2;

   TestTensor *output_reference = lstm_testdata.tensor_array + 3;

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

   lstm_seq( debug_arena, input->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             weights->data,
             biases->data,
             output_single_batch,
             layer_count
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
   res = load_testtensor(debug_arena, "testdata\\dw_conv_129.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\pw_conv_129_16.testtensor" );
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

TestResult v5_reparam_conv_test()
{
   MemoryArena *arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(arena, "testdata\\untracked\\v5_reparam_conv.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *reference_tensor = res.tensor_array + 3;

   ConvOutputShape output_shape = conv_output_shape_pad( input, weights, 1, 1 );

   size_t output_size = output_shape.batch_size * output_shape.channels_out * output_shape.sequence_length * sizeof( float );
   Assert( output_size == reference_tensor->nbytes );
   VAR_UNUSED( output_size );

   Assert(output_shape.batch_size == tdim(reference_tensor, 0));
   Assert(output_shape.channels_out == tdim(reference_tensor, 1));
   Assert(output_shape.sequence_length == tdim(reference_tensor, 2));

   TestTensor *input_padded = tensor_zero_pad_last_dim_lr(arena, input, 1, 1);
   TestTensor *output_tensor = conv_tensor_out(arena, input_padded, weights, biases, 1 );
   tensor_relu_inplace(output_tensor);


   float atol = 1e-4f;

   TestResult test_result = all_close( reference_tensor->data, output_tensor->data, reference_tensor->size, atol );

#if 0
   if ( test_result.max_error > atol )
   {
      //const char *funcname = __FUNCTION__;
      dump_tensor_hdr( "output.hdr", output_tensor );
      dump_tensor_hdr( "output_expected.hdr", reference_tensor );
   }
#endif

   endTemporaryMemory( mark );

   return test_result;
}

TestResult v5_reparam_conv2_test()
{
   MemoryArena *arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(arena, "testdata\\untracked\\v5_reparam_conv2.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *reference_tensor = res.tensor_array + 3;

   ConvOutputShape output_shape = conv_output_shape_pad( input, weights, 2, 1 );

   size_t output_size = output_shape.batch_size * output_shape.channels_out * output_shape.sequence_length * sizeof( float );
   Assert( output_size == reference_tensor->nbytes );
   VAR_UNUSED( output_size );

   Assert(output_shape.batch_size == tdim(reference_tensor, 0));
   Assert(output_shape.channels_out == tdim(reference_tensor, 1));
   Assert(output_shape.sequence_length == tdim(reference_tensor, 2));

   TestTensor *input_padded = tensor_zero_pad_last_dim_lr(arena, input, 1, 1);
   TestTensor *output_tensor = conv_tensor_out(arena, input_padded, weights, biases, 2 );
   tensor_relu_inplace(output_tensor);


   float atol = 1e-4f;

   TestResult test_result = all_close( reference_tensor->data, output_tensor->data, reference_tensor->size, atol );

#if 0
   if ( test_result.max_error > atol )
   {
      //const char *funcname = __FUNCTION__;
      dump_tensor_hdr( "output.hdr", output_tensor );
      dump_tensor_hdr( "output_expected.hdr", reference_tensor );
   }
#endif

   endTemporaryMemory( mark );

   return test_result;
}

TestResult v5_reparam_conv3_test()
{
   MemoryArena *arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(arena, "testdata\\untracked\\v5_reparam_conv3.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *reference_tensor = res.tensor_array + 3;

   ConvOutputShape output_shape = conv_output_shape_pad( input, weights, 2, 1 );

   size_t output_size = output_shape.batch_size * output_shape.channels_out * output_shape.sequence_length * sizeof( float );
   Assert( output_size == reference_tensor->nbytes );
   VAR_UNUSED( output_size );

   Assert(output_shape.batch_size == tdim(reference_tensor, 0));
   Assert(output_shape.channels_out == tdim(reference_tensor, 1));
   Assert(output_shape.sequence_length == tdim(reference_tensor, 2));

   TestTensor *input_padded = tensor_zero_pad_last_dim_lr(arena, input, 1, 1);
   TestTensor *output_tensor = conv_tensor_out(arena, input_padded, weights, biases, 2 );
   tensor_relu_inplace(output_tensor);


   float atol = 1e-4f;

   TestResult test_result = all_close( reference_tensor->data, output_tensor->data, reference_tensor->size, atol );

#if 0
   if ( test_result.max_error > atol )
   {
      //const char *funcname = __FUNCTION__;
      dump_tensor_hdr( "output.hdr", output_tensor );
      dump_tensor_hdr( "output_expected.hdr", reference_tensor );
   }
#endif

   endTemporaryMemory( mark );

   return test_result;
}

TestResult v5_reparam_conv4_test()
{
   MemoryArena *arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(arena, "testdata\\untracked\\v5_reparam_conv4.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *reference_tensor = res.tensor_array + 3;

   ConvOutputShape output_shape = conv_output_shape_pad( input, weights, 1, 1 );

   size_t output_size = output_shape.batch_size * output_shape.channels_out * output_shape.sequence_length * sizeof( float );
   Assert( output_size == reference_tensor->nbytes );
   VAR_UNUSED( output_size );

   Assert(output_shape.batch_size == tdim(reference_tensor, 0));
   Assert(output_shape.channels_out == tdim(reference_tensor, 1));
   Assert(output_shape.sequence_length == tdim(reference_tensor, 2));

   TestTensor *input_padded = tensor_zero_pad_last_dim_lr(arena, input, 1, 1);
   TestTensor *output_tensor = conv_tensor_out(arena, input_padded, weights, biases, 1 );
   tensor_relu_inplace(output_tensor);


   float atol = 1e-4f;

   TestResult test_result = all_close( reference_tensor->data, output_tensor->data, reference_tensor->size, atol );

#if 0
   if ( test_result.max_error > atol )
   {
      //const char *funcname = __FUNCTION__;
      dump_tensor_hdr( "output.hdr", output_tensor );
      dump_tensor_hdr( "output_expected.hdr", reference_tensor );
   }
#endif

   endTemporaryMemory( mark );

   return test_result;
}

TestResult first_layer_conv_block_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(debug_arena, "testdata\\first_layer_conv_block.testtensor" );
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

   conv_block( debug_arena, input, 1,
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
   res = load_testtensor(debug_arena, "testdata\\softmax_test.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\layernorm_test.testtensor" );
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

   layer_norm( debug_arena, input, weight, bias, output );

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
   res = load_testtensor(debug_arena, "testdata\\batchnorm_test.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\untracked\\stft_test.testtensor" );
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

   my_stft(debug_arena, input, forward_basis_buffer, output, 64, 128 );

   float atol = 1e-4f;
   TestResult test_result = all_close( result->data, output->data, result->size, atol );

   endTemporaryMemory( mark );

   return test_result;
}

TestResult stft_test_v5()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(debug_arena, "testdata\\untracked\\RED600_stft_v5.testtensor" );
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

   my_stft_(debug_arena, input, forward_basis_buffer, output, 128, 0, 64 );

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
   res = load_testtensor(debug_arena, "testdata\\adaptive_audio_normalization_test.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\dual_head_attention_test.testtensor" );
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

   dual_head_attention( debug_arena, input,
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
   res = load_testtensor(debug_arena, "testdata\\transformer_block_test_16_16_48.testtensor" );
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


TestResult transformer_first_layer_test()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(debug_arena, "testdata\\transformer_first_layer.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\transformer_layers_1_2.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\transformer_layers_1_2_3.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\transformer_layers_1_2_3_4.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\adaptive_normalization_encoder.testtensor" );
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
   res = load_testtensor(debug_arena, "testdata\\untracked\\stft_normalization_encoder.testtensor" );
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
   int half_filter_length;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input, forward_basis_buffer, 64, half_filter_length );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input, forward_basis_buffer, stft_output, 64, 128 );

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

   res = load_testtensor(debug_arena, "testdata\\untracked\\stft_normalization_encoder_lstm.testtensor" );
   lstm_weights_res = load_testtensor(debug_arena, "testdata\\untracked\\lstm_silero_3.1_16k_for_c.testtensor" );

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
   int half_filter_length;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input, forward_basis_buffer, 64, half_filter_length );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input, forward_basis_buffer, stft_output, 64, 128);

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

   lstm_seq( debug_arena, l4_output_t->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             lstm_weights->data,
             lstm_biases->data,
             output->data,
             layer_count
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

   res = load_testtensor(debug_arena, "testdata\\untracked\\stft_normalization_encoder_lstm_decoder.testtensor" );
   lstm_weights_res = load_testtensor(debug_arena, "testdata\\untracked\\lstm_silero_3.1_16k_for_c.testtensor" );

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
   int half_filter_length;
   {
      int filter_length = tdim( forward_basis_buffer, 2 );
      half_filter_length = filter_length / 2;
      cutoff = half_filter_length + 1;
   }
   // TODO(irwin): dehardcode 64 hop_length
   int stft_out_features_count = compute_stft_output_feature_count( input, forward_basis_buffer, 64, half_filter_length );
   TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input, -2 ), cutoff, stft_out_features_count );

   my_stft( debug_arena, input, forward_basis_buffer, stft_output, 64, 128 );

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

   lstm_seq( debug_arena, l4_output_t->data,
             seq_length * batches,
             input_size,
             input_h_array,
             input_c_array,
             lstm_weights->data,
             lstm_biases->data,
             lstm_output,
             layer_count
   );

   TestTensor *lstm_output_tensor = tensor_zeros_3d( debug_arena, batches, seq_length, input_size );
   memmove( lstm_output_tensor->data, lstm_output, lstm_output_tensor->nbytes );
   TestTensor *lstm_output_tensor_t = tensor_transpose_last_2d( debug_arena, lstm_output_tensor );

   int decoder_output_size = batches * tdim( decoder_weights, 0 );
   Assert( decoder_output_size == output->size );

   decoder_tensor(debug_arena, lstm_output_tensor_t, decoder_weights, decoder_biases, output );

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

   res = load_testtensor(debug_arena, "testdata\\untracked\\silero.testtensor" );
   silero_weights_res = load_testtensor(debug_arena, "testdata\\silero_v31_16k.testtensor" );

   if ( res.tensor_count == 0 || silero_weights_res.tensor_count == 0 )
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   int encoder_weights_count = 24 + 24 + 22 + 24;
   Assert( res.tensor_count == 2 );
   Assert( silero_weights_res.tensor_count == (1 + encoder_weights_count + 2 + 2) );

   Silero_Weights silero_weights = silero_weights_init( silero_weights_res );

   int test_data_index = 0;
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
      int half_filter_length;
      {
         int filter_length = tdim( silero_weights.forward_basis_buffer, 2 );
         half_filter_length = filter_length / 2;
         cutoff = half_filter_length + 1;
      }
      // TODO(irwin): dehardcode 64 hop_length
      int stft_out_features_count = compute_stft_output_feature_count( &input_one_batch, silero_weights.forward_basis_buffer, 64, half_filter_length );
      TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( &input_one_batch, -2 ), cutoff, stft_out_features_count );

      my_stft( debug_arena, &input_one_batch, silero_weights.forward_basis_buffer, stft_output, 64, 128 );

      TestTensor *normalization_output = tensor_copy( debug_arena, stft_output );
#endif
      adaptive_audio_normalization_inplace( debug_arena, normalization_output );

      ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, silero_weights.encoder_weights );
      TestTensor *l4_output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

      encoder( debug_arena, normalization_output, silero_weights.encoder_weights, l4_output );

      TestTensor *l4_output_t = tensor_transpose_last_2d( debug_arena, l4_output );

      int batches = tdim( l4_output_t, -3 );
      int seq_length = tdim( l4_output_t, -2 );
      int input_size = tdim( l4_output_t, -1 );
      int layer_count = tdim( silero_weights.lstm_weights, 0 );
      int hidden_size = tdim( silero_weights.lstm_weights, -1 ) / 2;
      Assert( hidden_size == input_size );
      Assert( hidden_size == tdim( silero_weights.lstm_biases, -1 ) / 4 );
      int batch_stride = seq_length * input_size;
      int lstm_output_size = batch_stride * batches + (input_size * layer_count * 2);

      int hc_size = input_size * layer_count;
      Assert( hc_size == lstm_input_h->size );

      float *lstm_output = pushArray( debug_arena, lstm_output_size, float );
      //float *lstm_output = pushArray( debug_arena, batches * seq_length * input_size, float );

      lstm_seq( debug_arena, l4_output_t->data,
                seq_length * batches,
                input_size,
                lstm_input_h->data,
                lstm_input_c->data,
                silero_weights.lstm_weights->data,
                silero_weights.lstm_biases->data,
                lstm_output,
                layer_count
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

      int decoder_output_size = batches * tdim( silero_weights.decoder_weights, 0 );

      int decoder_results = tdim( silero_weights.decoder_weights, 0 );
      TestTensor *output_decoder = tensor_zeros_3d( debug_arena, 1, decoder_results, 1 );
      Assert( decoder_output_size == output_decoder->size );

      decoder_tensor(debug_arena, lstm_output_tensor_t, silero_weights.decoder_weights, silero_weights.decoder_biases, output_decoder );

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
   res = load_testtensor(debug_arena, "testdata\\transformer_layers_3.testtensor" );
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

TestTensor *chunks_with_context_from_file(MemoryArena *arena, const char *path)
{
   TestTensor *chunks = 0;

   File_Contents contents = read_entire_file(arena, path);
   if (contents.contents != 0 && contents.bytes_count != 0)
   {
      s16 *pcm_s16le = (s16 *)contents.contents;
      int samples_count = (int)contents.bytes_count / 2;

      int context = 64;
      int window = 512;
      int chunk_size = context + window;
      // int overhead = samples_count / (512 / 64);

      float *samples_float = pushArray(arena, samples_count, float);
      for (int i = 0; i < samples_count; ++i)
      {
         samples_float[i] = (float)pcm_s16le[i] / 32768.0f;
      }

      int chunks_count = samples_count / window;
      int chunks_count_padded = chunks_count;
      if ((chunks_count * window) < samples_count)
      {
         ++chunks_count_padded;
      }

      chunks = tensor_zeros_2d(arena, chunks_count_padded, chunk_size);

      float *samples_cursor = samples_float + (window - context);
      float *tensor_cursor = chunks->data + context;

      memmove(tensor_cursor, samples_float, window * sizeof(float));
      tensor_cursor += window;

      for (int i = 1; i < chunks_count; ++i)
      {
         memmove(tensor_cursor, samples_cursor, chunk_size * sizeof(float));
         samples_cursor += window;

         tensor_cursor += chunk_size;
      }

      // NOTE(irwin): remainder
      if (chunks_count_padded != chunks_count)
      {
         float *one_past_last_sample = samples_float + samples_count;
         s64 padded_chunk_sample_size_in_bytes = one_past_last_sample - samples_cursor;

         memmove(tensor_cursor, samples_cursor, padded_chunk_sample_size_in_bytes);
      }
   }

   return chunks;
}

TestResult silero_v5_test()
{
   MemoryArena *arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( arena );

   LoadTesttensorResult res = {0};
   res = load_testtensor(arena, "testdata\\untracked\\RED600_silero_v5.testtensor" );
   if (res.tensor_count == 0)
   {
      endTemporaryMemory( mark );
      TestResult test_result = {0};
      return test_result;
   }

   // TODO(irwin): validate loaded tensor count helpers
   Assert( res.tensor_count == 15 );

   int test_data_index = 0;
#if 0
   TestTensor *input = res.tensor_array + test_data_index++;
#else
   test_data_index++;
   TestTensor *input = chunks_with_context_from_file(arena, "RED.s16le");
#endif

   TestTensor *forward_basis_buffer = res.tensor_array + test_data_index++;

   TestTensor *reparam_conv_0_weights = res.tensor_array + test_data_index++;
   TestTensor *reparam_conv_0_biases = res.tensor_array + test_data_index++;

   TestTensor *reparam_conv_1_weights = res.tensor_array + test_data_index++;
   TestTensor *reparam_conv_1_biases = res.tensor_array + test_data_index++;

   TestTensor *reparam_conv_2_weights = res.tensor_array + test_data_index++;
   TestTensor *reparam_conv_2_biases = res.tensor_array + test_data_index++;

   TestTensor *reparam_conv_3_weights = res.tensor_array + test_data_index++;
   TestTensor *reparam_conv_3_biases = res.tensor_array + test_data_index++;

   TestTensor *lstm_weights = res.tensor_array + test_data_index++;
   TestTensor *lstm_biases = res.tensor_array + test_data_index++;

   TestTensor *decoder_weights = res.tensor_array + test_data_index++;
   TestTensor *decoder_biases = res.tensor_array + test_data_index++;


   TestTensor *reference_probs = res.tensor_array + test_data_index++;

   TestTensor *result_probs = tensor_zeros_like(arena, reference_probs);


   // int input_size = tdim(input, 2);
   int input_size = tdim(lstm_biases, -1) / 4;
   int layer_count = tdim(lstm_weights, 0);
   // int hc_size = input_size * layer_count;
   TestTensor *lstm_hn = tensor_zeros_2d(arena, layer_count, input_size);
   TestTensor *lstm_cn = tensor_zeros_2d(arena, layer_count, input_size);


   int chunks_count = tdim(input, 0);
   int batch_size = 96;

   for (int batch_index = 0; batch_index < chunks_count; batch_index += batch_size)
   {
      int to = (batch_index + batch_size) - 1;
      if (to >= tdim(input, 0))
      {
         to = tdim(input, 0) - 1;
      }

      TestTensor stft_input_ = tensor_slice_first_dim(input, batch_index, to);
      TestTensor *stft_input = &stft_input_;

      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): STFT
      ////////////////////////////////////////////////////////////////////////////
      TestTensor *stft_output = 0;

      int hop_length = 128;
      int pad_left = 0;
      int pad_right = 64;
      {
         int filter_length = tdim(forward_basis_buffer, 2);
         int half_filter_length = filter_length / 2;
         int cutoff = half_filter_length + 1;

         int features_count = compute_stft_output_feature_count_lr( stft_input, forward_basis_buffer, hop_length, pad_left, pad_right );

         stft_output = tensor_zeros_3d( arena, tdim(stft_input, 0), cutoff, features_count );
      }


      my_stft_(arena, stft_input, forward_basis_buffer, stft_output, hop_length, pad_left, pad_right );

      TestTensor *reparam_conv_0_output = 0;
      TestTensor *reparam_conv_1_output = 0;
      TestTensor *reparam_conv_2_output = 0;
      TestTensor *reparam_conv_3_output = 0;

      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): Encoder.0
      ////////////////////////////////////////////////////////////////////////////
      {
         TestTensor *stft_output_padded = tensor_zero_pad_last_dim_lr(arena, stft_output, 1, 1);
         reparam_conv_0_output = conv_tensor_out(arena, stft_output_padded, reparam_conv_0_weights, reparam_conv_0_biases, 1 );
         tensor_relu_inplace(reparam_conv_0_output);
      }

      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): Encoder.1
      ////////////////////////////////////////////////////////////////////////////
      {
         TestTensor *reparam_conv_0_output_padded = tensor_zero_pad_last_dim_lr(arena, reparam_conv_0_output, 1, 1);
         reparam_conv_1_output = conv_tensor_out(arena, reparam_conv_0_output_padded, reparam_conv_1_weights, reparam_conv_1_biases, 2 );
         tensor_relu_inplace(reparam_conv_1_output);
      }

      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): Encoder.2
      ////////////////////////////////////////////////////////////////////////////
      {
         TestTensor *reparam_conv_1_output_padded = tensor_zero_pad_last_dim_lr(arena, reparam_conv_1_output, 1, 1);
         reparam_conv_2_output = conv_tensor_out(arena, reparam_conv_1_output_padded, reparam_conv_2_weights, reparam_conv_2_biases, 2 );
         tensor_relu_inplace(reparam_conv_2_output);
      }

      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): Encoder.3
      ////////////////////////////////////////////////////////////////////////////
      {
         TestTensor *reparam_conv_2_output_padded = tensor_zero_pad_last_dim_lr(arena, reparam_conv_2_output, 1, 1);
         reparam_conv_3_output = conv_tensor_out(arena, reparam_conv_2_output_padded, reparam_conv_3_weights, reparam_conv_3_biases, 1 );
         tensor_relu_inplace(reparam_conv_3_output);
      }

      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): LSTM
      ////////////////////////////////////////////////////////////////////////////
      TestTensor *reparam_conv_3_output_t = tensor_transpose_last_2d(arena, reparam_conv_3_output);

      LSTM_Result lstm_out = lstm_tensor_minibatched(arena, reparam_conv_3_output_t, lstm_weights, lstm_biases, lstm_hn, lstm_cn);
      memmove(lstm_hn->data, lstm_out.hn.data, lstm_out.hn.nbytes);
      memmove(lstm_cn->data, lstm_out.cn.data, lstm_out.cn.nbytes);

      TestTensor *lstm_out_tensor_t = tensor_transpose_last_2d(arena, &lstm_out.output);


      ////////////////////////////////////////////////////////////////////////////
      // NOTE(irwin): decoder
      ////////////////////////////////////////////////////////////////////////////
      TestTensor *decoder_out = 0;
      {

         TestTensor *relu_out = tensor_copy(arena, lstm_out_tensor_t);
         tensor_relu_inplace(relu_out);

         decoder_out = conv_tensor_out(arena, relu_out, decoder_weights, decoder_biases, 1);
         mysigmoid_inplace(decoder_out->data, decoder_out->size);
      }

      memmove(result_probs->data + batch_index, decoder_out->data, decoder_out->nbytes);
   }

   float atol = 1e-4f;
   TestResult test_result = all_close( reference_probs->data, result_probs->data, reference_probs->size, atol );

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
   TEST_FUNCTION_DESCRIPTION(lstm_test_RED_new),
   TEST_FUNCTION_DESCRIPTION(lstm_test_RED_1layer),

   TEST_FUNCTION_DESCRIPTION(stft_test_v5),
   TEST_FUNCTION_DESCRIPTION(v5_reparam_conv_test),
   TEST_FUNCTION_DESCRIPTION(v5_reparam_conv2_test),
   TEST_FUNCTION_DESCRIPTION(v5_reparam_conv3_test),
   TEST_FUNCTION_DESCRIPTION(v5_reparam_conv4_test),
   TEST_FUNCTION_DESCRIPTION(lstm_test_RED_v5),
   TEST_FUNCTION_DESCRIPTION(decoder_test_v5),
   TEST_FUNCTION_DESCRIPTION(silero_v5_test),
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
