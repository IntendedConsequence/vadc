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
static inline void convolve_k5_pad2 ( const float *arr, int count, const float *kernel_flipped, float *arr_out, float bias )
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

// NOTE(irwin): batch support almost ready
// but filters are hardcoded to 2 dims
// usually filters for conv1d are in the shape (out_channels, in_channels/groups, kernel_size)
// (where out_channels == filter_count)
// but for dw_conv, in_channels/groups == 1 and it's squeezed out, leaving (out_channels, kernel_size)
static void dw_conv_tensor ( TestTensor *input, TestTensor *filters, TestTensor *biases, TestTensor *output )
{
   TracyCZone(dw_conv_tensor, true);

   Assert( tensor_is_valid( input ) );
   Assert( tensor_is_valid( filters ) );
   Assert( tensor_is_valid( biases ) );
   Assert( tensor_is_valid( output ) );

   // TODO(irwin): for batch support
   // - Assert ndim == 3
   // - batch_size = tdim(input, 0)
   Assert( input->ndim == 2 || input->ndim == 3 );
   int batch_size = 1;

   Assert( filters->ndim == 2 || (filters->ndim == 3 && tdim(filters, -2) == 1) );
   Assert( biases->ndim == 1 );
   Assert( output->ndim == input->ndim );


   int sequence_length_in = tdim(input, -1);
   int in_channels = tdim(input, -2);
   int out_channels = tdim(filters, 0);

   int in_out_channels_groups = in_channels;

   Assert( out_channels == in_out_channels_groups );
   Assert( tdim(biases, 0) == out_channels );
   Assert( tdim(output, -2) == out_channels );

   int batch_stride = input->size / batch_size;
   int filter_len = tdim( filters, -1 );

   for (int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      int batch_offset = batch_index * batch_stride;
      for ( int i = 0; i < in_out_channels_groups; ++i )
      {
         float *arr_in = input->data + batch_offset + i * sequence_length_in;
         float *arr_out = output->data + batch_offset + i * sequence_length_in;

         float *arr_filters = filters->data + i * filter_len;
         float bias = biases->data[i];
         convolve_k5_pad2( arr_in, sequence_length_in, arr_filters, arr_out, bias );
      }
   }

   TracyCZoneEnd(dw_conv_tensor);
}

static inline void conv_tensor ( TestTensor *input, TestTensor *filters, TestTensor *biases, int hop_length, TestTensor *output )
{
   TracyCZone(conv_tensor, true);

   Assert( tensor_is_valid( input ) );
   Assert( tensor_is_valid(filters ) );
   if (biases)
   {
      Assert( tensor_is_valid( biases ) );
   }
   Assert( tensor_is_valid( output ) );

   Assert(input->ndim == 2 || input->ndim == 3);
   Assert(output->ndim == 2 || output->ndim == 3);

   int batch_size;
   int in_channels;
   if ( input->ndim == 2 )
   {
      batch_size = 1;
      in_channels = tdim(input, 0);
   }
   else
   {
      batch_size = tdim(input, 0);
      in_channels = tdim(input, 1);
   }
   int array_count = tdim(input, -1);

   if (batch_size != 1)
   {
      Assert(output->ndim == 3);
      Assert(tdim(output, 0) == batch_size);
   }


   Assert( filters->ndim == 3 );

   int out_channels = filters->dims[0];
   int filter_count = out_channels;
   int kernel_size = filters->dims[2];
   int output_array_count = 1 + (array_count - kernel_size) / hop_length;

   Assert( filters->dims[1] == in_channels );
   Assert( tdim(output, -2) == filter_count );
   Assert( tdim(output, -1) == output_array_count );
   if (biases)
   {
      Assert( biases->ndim == 1 );
      Assert( biases->dims[0] == filter_count );
   }

   int batch_stride_input = input->size / batch_size;
   int batch_stride_output = output->size / batch_size;

   // float *to_sum = pushArray(DEBUG_getDebugArena(), kernel_size, float);

   for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      float *input_data_batch = input->data + batch_index * batch_stride_input;
      float *output_data_batch = output->data + batch_index * batch_stride_output;
      for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
      {
         float *output_filter_channel = output_data_batch + filter_index * output_array_count;
         if (biases)
         {
            float bias_value = biases->data[filter_index];
            for (int i = 0; i < output_array_count; ++i)
            {
               output_filter_channel[i] = bias_value;
            }
         }

         for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
         {
            float *kernel = index3d( filters, filter_index, channel_index, 0 );

            float *channel = input_data_batch + channel_index * array_count;
            for ( int index = 0; index < output_array_count; ++index )
            {
               #if 0
               output_filter_channel[index] += dotproduct_slow( channel + index * hop_length, kernel_size, kernel, kernel_size );
               #elif 0
               float *channel_sub = channel + index * hop_length;
               float out_channel = output_filter_channel[index];

               __m256 s = _mm256_setzero_ps();
               int i;
               for (i = 0; i < kernel_size - 7; i += 8)
               {
                  __m256 a = _mm256_loadu_ps(channel_sub + i);
                  __m256 b = _mm256_loadu_ps(kernel + i);
                  __m256 r = _mm256_mul_ps(a, b);
                  s = _mm256_add_ps(s, r);
               }

               s = _mm256_hadd_ps(s, s);
               s = _mm256_hadd_ps(s, s);
               s = _mm256_hadd_ps(s, s);

               out_channel += ((float *)&s)[0];
               // float v;
               // _MM_EXTRACT_FLOAT(v, _mm256_extractf128_ps(s, 0), 0);
               // out_channel += v;


               for (; i < kernel_size; ++i)
               {
                  out_channel += channel_sub[i] * kernel[i];
               }

               output_filter_channel[index] = out_channel;
               #else
               int wide = 8;
               int wide_parts = kernel_size / wide;
               float sub = 0.0f;
               float *channel_sub = channel + index * hop_length;
               for (int i = 0; i < wide_parts; ++i)
               {
                  float *channel_sub_sub = channel_sub + i * wide;
                  float *kernel_sub = kernel + i * wide;

                  float subsub = 0.0f;
                  for (int j = 0; j < wide; ++j)
                  {
                     subsub += channel_sub_sub[j] * kernel_sub[j];
                  }
                  sub += subsub;
               }

               float sub2 = 0.0f;
               for (int i = wide_parts * wide; i < kernel_size; ++i)
               {
                  float vala = channel_sub[i];
                  float valb = kernel[i];
                  float muled = vala * valb;
                  float added = sub2 + muled;
                  sub2 = added;
               }

               float sum = sub + sub2;
               float read = output_filter_channel[index];
               output_filter_channel[index] = sum + read;

               #endif
            }
         }
      }
   }

   TracyCZoneEnd(conv_tensor);
}


static inline TestTensor *conv_tensor_out ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *biases, int hop_length )
{
   // TracyCZone(conv_tensor_out, true);

   TestTensor *output = tensor_zeros_for_conv( arena, input, filters, hop_length );
   conv_tensor( input, filters, biases, hop_length, output );

   // TracyCZoneEnd(conv_tensor_out);
   return output;
}

static inline void pw_conv_tensor ( TestTensor *input, TestTensor *filters, TestTensor *biases, TestTensor *output )
{
   TracyCZone(pw_conv_tensor, true);

   conv_tensor( input, filters, biases, 1, output );

   TracyCZoneEnd(pw_conv_tensor);
}

static inline TestTensor *pw_conv_tensor_out ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *biases )
{
   TracyCZone(pw_conv_tensor_out, true);
   return conv_tensor_out( arena, input, filters, biases, 1 );

   TracyCZoneEnd(pw_conv_tensor_out);
}


static void conv_tensor_stride64_nobias ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *output )
{
   TracyCZone(conv_tensor_stride64_nobias, true);

   TemporaryMemory mark = beginTemporaryMemory( arena );

   // int mock_biases_dims[1] = { filters->dims[0] };
   // TestTensor *biases = tensor_zeros( arena, ArrayCount(mock_biases_dims), mock_biases_dims );

   conv_tensor( input, filters, NULL, 64, output );

   endTemporaryMemory( mark );
   TracyCZoneEnd(conv_tensor_stride64_nobias);
}



static void conv_block(MemoryArena *arena,  TestTensor *input, b32 has_out_proj,
                        TestTensor *dw_weights, TestTensor *dw_biases,
                        TestTensor *pw_weights, TestTensor *pw_biases,
                        TestTensor *proj_weights, TestTensor *proj_biases,
                        TestTensor *output )
{
   TracyCZone(conv_block, true);


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

   MemoryArena *debug_arena = arena;
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   //TestTensor *dw_output = tensor_zeros_2d( debug_arena, input->dims[0], input->dims[1] );
   TestTensor *dw_output = tensor_zeros_like( debug_arena, input );

   dw_conv_tensor( input, dw_weights, dw_biases, dw_output );
   tensor_relu_inplace( dw_output );

   TestTensor *pw_output = output;
   // TestTensor *pw_output = tensor_zeros_2d(debug_arena, pw_output_dims[0], pw_output_dims[1]);
   pw_conv_tensor( dw_output, pw_weights, pw_biases, pw_output );

   if ( has_out_proj )
   {
      TestTensor *out_proj = tensor_zeros_like( debug_arena, pw_output );
      pw_conv_tensor( input, proj_weights, proj_biases, out_proj );

      add_arrays_inplace( pw_output->data, pw_output->size, out_proj->data );
   }
   else
   {
      add_arrays_inplace( pw_output->data, pw_output->size, input->data );
   }

   tensor_relu_inplace( output );



   endTemporaryMemory( mark );

   TracyCZoneEnd(conv_block);
}
