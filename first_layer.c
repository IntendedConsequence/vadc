#include "utils.h"
#include "memory.h"
#include "tensor.h"
#include "maths.h"

// NOTE(irwin): specialized for kernel_size = 5, padding = 2 (with zeros)
// IMPORTANT(irwin): apparently, expecting weights with flipped kernels
// coincidentally matches PyTorch conv1d implementation, where conv1d is
// implemented actually as cross-correlation. Since convolution is
// cross correlation with weight kernels flipped, PyTorch convolution really
// is cross correlation. Numpy's np.convolve however does flip the kernels,
// so watch out for that.
// It seems that the reason for PyTorch conv1d being this way is just because
// when training a CNN, convolution and cross correlation don't have effect
// on training, but the extra step of flipping the kernels just slows down
// the process unnecessarily.
VADC_API
void convolve_k5_pad2 ( const float *arr, int count, const float *kernel_flipped, float *arr_out, float bias )
{
   int kernel_size = 5;
   int padding = 2;
   // NOTE(irwin): for kernel_size = 5, padding = 2, out_array_count equals count
   int out_array_count = count - kernel_size + 1 + padding + padding;

   // NOTE(irwin): since we know that padding is 2 zeros, we can compute first two elements as if we had a kernel
   // of size 4 and 3 for elements at index 0 and 1, respectively, because the padded zeroes effectively mask out
   // the first elements of the kernel.
   arr_out[0] = bias + dotproduct( arr, kernel_size - 2, kernel_flipped + 2, kernel_size - 2 );
   arr_out[1] = bias + dotproduct( arr, kernel_size - 1, kernel_flipped + 1, kernel_size - 1 );

   for ( int i = 0; i < count - kernel_size + 1; ++i )
   {
      float value = dotproduct( arr + i, kernel_size, kernel_flipped, kernel_size );
      arr_out[padding + i] = bias + value;
   }

   // NOTE(irwin): we repeat the same thing for the last two elements as we did for the first two. However,
   // this would mean we need to get the pointer to the last 4 and 3 elements of the input array. This would
   // mean `arr + count - 4` and `arr + count - 3`, or `arr + count - kernel_size + 1`. BUT!
   // If we did that, the calls to dotproduct would look like:
   //
   // ... = dotproduct(arr_pad + 0, kernel_size - 1, kernel_flipped, kernel_size - 1);
   // ... = dotproduct(arr_pad + 1, kernel_size - 2, kernel_flipped, kernel_size - 2);
   // which is harder to read and understand, which offsets do what, from which end etc
   // So to make it more uniform, we instead compute the pointer to the last kernel_size elements of the array,
   // so the offsets are matched now.
   // We do the same thing with arr_out_one_before_two_last_elements following the same principle, with the only
   // difference being we get the pointer to one output array element BEFORE the last two output elements,
   // which we can then offset by the same amount.
   const float *arr_pad = arr + count - kernel_size;
   float *arr_out_one_before_two_last_elements = arr_out + out_array_count - 2 - 1;
   arr_out_one_before_two_last_elements[1] = bias + dotproduct( arr_pad + 1, kernel_size - 1, kernel_flipped, kernel_size - 1 );
   arr_out_one_before_two_last_elements[2] = bias + dotproduct( arr_pad + 2, kernel_size - 2, kernel_flipped, kernel_size - 2 );
}

static void dw_conv_tensor ( TestTensor input, int in_out_channels_groups, TestTensor filters, TestTensor biases, TestTensor output )
{
   Assert( tensor_is_valid( input ) );
   Assert( tensor_is_valid( filters ) );
   Assert( tensor_is_valid( biases ) );
   Assert( tensor_is_valid( output ) );

   Assert( input.ndim == 2 );
   Assert( filters.ndim == 2 );
   Assert( biases.ndim == 1 );
   Assert( output.ndim == 2 );

   Assert( input.dims[0] == in_out_channels_groups );
   Assert( filters.dims[0] == in_out_channels_groups );
   Assert( biases.dims[0] == in_out_channels_groups );
   Assert( output.dims[0] == in_out_channels_groups );

   for ( int i = 0; i < in_out_channels_groups; ++i )
   {
      float *arr_out = index2d( output, i, 0 );
      float *arr_in = index2d( input, i, 0 );
      float *arr_filters = index2d( filters, i, 0 );
      float bias = biases.data[i];
      convolve_k5_pad2( arr_in, input.dims[1], arr_filters, arr_out, bias );
   }
}

static void pw_conv_tensor ( TestTensor input, TestTensor filters, TestTensor biases, TestTensor output )
{
   Assert( tensor_is_valid( input ) );
   Assert( tensor_is_valid( filters ) );
   Assert( tensor_is_valid( biases ) );
   Assert( tensor_is_valid( output ) );

   Assert( input.ndim == 2 );
   Assert( filters.ndim == 3 );
   Assert( biases.ndim == 1 );
   Assert( output.ndim == 2 );

   int in_channels = input.dims[0];
   int array_count = input.dims[1];
   int out_channels = filters.dims[0];
   int filter_count = out_channels;
   int kernel_size = filters.dims[2];
   int output_array_count = array_count - kernel_size + 1;

   Assert( filters.dims[1] == in_channels );
   Assert( output.dims[0] == filter_count );
   Assert( output.dims[1] == output_array_count );
   Assert( biases.dims[0] == filter_count );

   for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
   {

      TestTensor output_filter = tensor_slice_first_dim( output, filter_index );
      broadcast_value_to_tensor( output_filter, biases.data[filter_index] );
      // zero_tensor(output_filter);
      // float *output_arr = index2d(output, filter_index, 0);

      // memset(output_arr, 0, output_array_count * sizeof(float));
      for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
      {
         float *kernel = index3d( filters, filter_index, channel_index, 0 );

         float *channel = index2d( input, channel_index, 0 );
         for ( int index = 0; index < array_count; ++index )
         {
            output_filter.data[index] += dotproduct( channel + index, kernel_size, kernel, kernel_size );
         }
      }
   }
}



static void conv_block( TestTensor input, int in_channels, int out_channels_pw_proj, b32 has_out_proj,
                        TestTensor dw_weights, TestTensor dw_biases,
                        TestTensor pw_weights, TestTensor pw_biases,
                        TestTensor proj_weights, TestTensor proj_biases,
                        TestTensor output )
{
   VAR_UNUSED( out_channels_pw_proj );

   Assert( tensor_is_valid( input ) );
   Assert( tensor_is_valid( dw_weights ) );
   Assert( tensor_is_valid( dw_biases ) );
   Assert( tensor_is_valid( pw_weights ) );
   Assert( tensor_is_valid( pw_biases ) );
   if ( has_out_proj )
   {
      Assert( tensor_is_valid( proj_weights ) );
      Assert( tensor_is_valid( proj_biases ) );
   }
   Assert( tensor_is_valid( output ) );

   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   TestTensor *dw_output = tensor_zeros_2d( debug_arena, input.dims[0], input.dims[1] );

   dw_conv_tensor( input, in_channels, dw_weights, dw_biases, *dw_output );
   tensor_relu_inplace( *dw_output );

   // NOTE(irwin): pw_output size calc
   int pw_output_dims[] = {0, 0};
   {
      int array_count = dw_output->dims[1];
      int out_channels = pw_weights.dims[0];
      int filter_count = out_channels;
      int kernel_size = pw_weights.dims[2];
      int output_array_count = array_count - kernel_size + 1;

      pw_output_dims[0] = filter_count;
      pw_output_dims[1] = output_array_count;
   }

   TestTensor *pw_output = &output;
   // TestTensor *pw_output = tensor_zeros_2d(debug_arena, pw_output_dims[0], pw_output_dims[1]);
   pw_conv_tensor( *dw_output, pw_weights, pw_biases, *pw_output );

   if ( has_out_proj )
   {
      TestTensor *out_proj = tensor_zeros_like( debug_arena, pw_output );
      pw_conv_tensor( input, proj_weights, proj_biases, *out_proj );

      add_arrays_inplace( pw_output->data, pw_output->size, out_proj->data );
   }

   tensor_relu_inplace( output );



   endTemporaryMemory( mark );
}