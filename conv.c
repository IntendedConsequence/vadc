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
   // - [ ] Assert ndim == 3
   // - [x] batch_size = tdim(input, 0)
   Assert( input->ndim == 2 || input->ndim == 3 );

   int batch_size = 1;
   if (input->ndim == 3)
   {
      batch_size = tdim(input, 0);
   }

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

   if (kernel_size == 1 && hop_length == 1)
   {
      MemoryArena *arena = DEBUG_getDebugArena();

      for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
      {
         float *input_data_batch = input->data + batch_index * batch_stride_input;
         float *output_data_batch = output->data + batch_index * batch_stride_output;

         for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
         {
            TemporaryMemory batch_mark = beginTemporaryMemory(arena);

            float bias_value = 0.0f;
            if (biases)
            {
               bias_value = biases->data[filter_index];
            }

            float *temp = pushArray(arena, batch_stride_input, float);

#if 0
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): variant A with premultiplied channels by the kernels
            /////////////////////////////////////////////////////////////////////////////

            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): shared prologue
            //              multiply all channels by the kernels into a temp array
            /////////////////////////////////////////////////////////////////////////////

            for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
            {

               float *in_channel = input_data_batch + channel_index * array_count;
               float *out_channel = temp + channel_index * array_count;

               float kernel = *index3d( filters, filter_index, channel_index, 0 );

               for (int i = 0; i < array_count; ++i)
               {
                  out_channel[i] = in_channel[i] * kernel;
               }
            }
# if 1
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): subvariant A1:
            //              sum using binary map-reduce
            /////////////////////////////////////////////////////////////////////////////

            int size = in_channels;
            while (size > 1)
            {
               int half = size / 2;
               for ( int channel_index = 0; channel_index < half; ++channel_index )
               {
                  int left_index = channel_index;
                  int right_index = channel_index + half;

                  float *channel_left = temp + left_index * array_count;
                  float *channel_right = temp + right_index * array_count;

                  for (int i = 0; i < array_count; ++i)
                  {
                     channel_left[i] += channel_right[i];
                  }
               }

               b32 is_odd = size % 2 > 0;
               if (is_odd)
               {
                  int left_index = half - 1;
                  int right_index = size - 1;

                  float *channel_left = temp + left_index * array_count;
                  float *channel_right = temp + right_index * array_count;

                  for (int i = 0; i < array_count; ++i)
                  {
                     channel_left[i] += channel_right[i];
                  }
               }

               size = half;
            }

# else
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): subvariant A3:
            //              naive sequential sum
            /////////////////////////////////////////////////////////////////////////////

            for (int i = 0; i < array_count; ++i)
            {
               for (int channel_index = 1; channel_index < in_channels; ++channel_index)
               {
                  float *channel_right = temp + channel_index * array_count;
                  temp[i] += channel_right[i];
               }
            }

# endif
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): shared bias addition at the end
            /////////////////////////////////////////////////////////////////////////////

            float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            for (int i = 0; i < array_count; ++i)
            {
               output_filter_channel[i] = temp[i] + bias_value;
            }
#elif 0
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): variant B:
            //              in_channels are transposed, then multiplied by the kernels
            //              and summed up with binary divide and conquer map-reduce
            /////////////////////////////////////////////////////////////////////////////

            // NOTE(irwin): write transposed
            for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
            {

               float *in_channel = input_data_batch + channel_index * array_count;

               for (int i = 0; i < array_count; ++i)
               {
                  float *out_channel = temp + in_channels * i;
                  out_channel[channel_index] = in_channel[i];
               }
            }

            float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            for (int i = 0; i < array_count; ++i)
            {
               int j = 0;
               int stride = 16;
               for (; j < in_channels - (stride - 1); j += stride)
               {
                  float *kernels = index3d( filters, filter_index, j + 0, 0 );
                  float k0 = kernels[0];
                  float k1 = kernels[1];
                  float k2 = kernels[2];
                  float k3 = kernels[3];
                  float k4 = kernels[4];
                  float k5 = kernels[5];
                  float k6 = kernels[6];
                  float k7 = kernels[7];
                  float k8 = kernels[8];
                  float k9 = kernels[9];
                  float k10 = kernels[10];
                  float k11 = kernels[11];
                  float k12 = kernels[12];
                  float k13 = kernels[13];
                  float k14 = kernels[14];
                  float k15 = kernels[15];

                  int transposed_stride = i * in_channels;
                  float *row = temp + transposed_stride + j;
                  float a0  = row[ 0];
                  float a1  = row[ 1];
                  float a2  = row[ 2];
                  float a3  = row[ 3];
                  float a4  = row[ 4];
                  float a5  = row[ 5];
                  float a6  = row[ 6];
                  float a7  = row[ 7];
                  float a8  = row[ 8];
                  float a9  = row[ 9];
                  float a10 = row[10];
                  float a11 = row[11];
                  float a12 = row[12];
                  float a13 = row[13];
                  float a14 = row[14];
                  float a15 = row[15];

                  float a0a1   =  a0 * k0 +  a1 * k1;
                  float a2a3   =  a2 * k2 +  a3 * k3;
                  float a4a5   =  a4 * k4 +  a5 * k5;
                  float a6a7   =  a6 * k6 +  a7 * k7;
                  float a8a9   =  a8 * k8 +  a9 * k9;
                  float a10a11 = a10 * k10 + a11 * k11;
                  float a12a13 = a12 * k12 + a13 * k13;
                  float a14a15 = a14 * k14 + a15 * k15;

                  float a0a1a2a3 = a0a1 + a2a3;
                  float a4a5a6a7 = a4a5 + a6a7;
                  float a8a9a10a11 = a8a9 + a10a11;
                  float a12a13a14a15 = a12a13 + a14a15;

                  float a0a1a2a3a4a5a6a7 = a0a1a2a3 + a4a5a6a7;
                  float a8a9a10a11a12a13a14a15 = a8a9a10a11 + a12a13a14a15;

                  output_filter_channel[i] += (a0a1a2a3a4a5a6a7 + a8a9a10a11a12a13a14a15);
               }
               for (; j < in_channels; ++j)
               {
                  int transposed_stride = i * in_channels;
                  float k0 = *index3d( filters, filter_index, j + 0, 0 );
                  float a0 = temp[transposed_stride + j + 0] * k0;
                  output_filter_channel[i] += (a0);
               }

               output_filter_channel[i] += bias_value;
            }
#elif 0
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): variant C:
            //              in_channels are premultiplied by the kernels and written
            //              transposed, then summed up N at a time where N is a stride of
            //              2-16 (power of 2)
            /////////////////////////////////////////////////////////////////////////////
            for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
            {

               float *in_channel = input_data_batch + channel_index * array_count;

               float kernel = *index3d( filters, filter_index, channel_index, 0 );

               for (int i = 0; i < array_count; ++i)
               {
                  float *out_channel = temp + in_channels * i;
                  out_channel[channel_index] = in_channel[i] * kernel;
               }
            }
            float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            for (int i = 0; i < array_count; ++i)
            {
               int j = 0;
               #define STRIDE 8
               int stride = STRIDE;
               for (; j < in_channels - (stride - 1); j += stride)
               {
                  int transposed_stride = i * in_channels;
                  float a0 = temp[transposed_stride + j + 0];
                  float a1 = temp[transposed_stride + j + 1];
                  float a2 = temp[transposed_stride + j + 2];
                  float a3 = temp[transposed_stride + j + 3];
                  #if STRIDE > 4
                  float a4 = temp[transposed_stride + j + 4];
                  float a5 = temp[transposed_stride + j + 5];
                  float a6 = temp[transposed_stride + j + 6];
                  float a7 = temp[transposed_stride + j + 7];
                  #endif
                  #if STRIDE > 8
                  float a8 = temp[transposed_stride + j + 8];
                  float a9 = temp[transposed_stride + j + 9];
                  float a10 = temp[transposed_stride + j + 10];
                  float a11 = temp[transposed_stride + j + 11];
                  float a12 = temp[transposed_stride + j + 12];
                  float a13 = temp[transposed_stride + j + 13];
                  float a14 = temp[transposed_stride + j + 14];
                  float a15 = temp[transposed_stride + j + 15];
                  #endif
                  output_filter_channel[i] += (
                                               (
                                                ((a0 + a1) + (a2 + a3))
                  #if STRIDE > 4
                                             +  ((a4 + a5) + (a6 + a7))
                  #endif
                                               )
                  #if STRIDE > 8
                                             + (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15)))
                  #endif
                                              );
               }
               #undef STRIDE
               for (; j < in_channels; ++j)
               {
                  int transposed_stride = i * in_channels;
                  float a0 = temp[transposed_stride + j + 0];
                  output_filter_channel[i] += (a0);
               }

               output_filter_channel[i] += bias_value;
            }
#elif 0
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): variant D: like C but explicit transpose step
            //              in_channels are premultiplied by the kernels
            //              then summed up N at a time where N is a stride of
            //              2-16 (power of 2)
            /////////////////////////////////////////////////////////////////////////////
            float *temp2 = pushArray(arena, batch_stride_input, float);

            for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
            {

               float *in_channel = input_data_batch + channel_index * array_count;
               float *out_channel = temp + channel_index * array_count;

               float kernel = *index3d( filters, filter_index, channel_index, 0 );

               for (int i = 0; i < array_count; ++i)
               {
                  out_channel[i] = in_channel[i] * kernel;
               }
            }

            // NOTE(irwin): transpose
            for ( int y = 0; y < in_channels; ++y )
            {
               for (int x = 0; x < array_count; ++x)
               {
                  int left = array_count * y + x;
                  int right = in_channels * x + y;

                  temp2[right] = temp[left];
               }
            }

            float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            for (int i = 0; i < array_count; ++i)
            {
               int j = 0;
               #define STRIDE 8
               int stride = STRIDE;
               for (; j < in_channels - (stride - 1); j += stride)
               {
                  int transposed_stride = i * in_channels;
                  float a0 = temp2[transposed_stride + j + 0];
                  float a1 = temp2[transposed_stride + j + 1];
                  float a2 = temp2[transposed_stride + j + 2];
                  float a3 = temp2[transposed_stride + j + 3];
                  #if STRIDE > 4
                  float a4 = temp2[transposed_stride + j + 4];
                  float a5 = temp2[transposed_stride + j + 5];
                  float a6 = temp2[transposed_stride + j + 6];
                  float a7 = temp2[transposed_stride + j + 7];
                  #endif
                  #if STRIDE > 8
                  float a8 = temp2[transposed_stride + j + 8];
                  float a9 = temp2[transposed_stride + j + 9];
                  float a10 = temp2[transposed_stride + j + 10];
                  float a11 = temp2[transposed_stride + j + 11];
                  float a12 = temp2[transposed_stride + j + 12];
                  float a13 = temp2[transposed_stride + j + 13];
                  float a14 = temp2[transposed_stride + j + 14];
                  float a15 = temp2[transposed_stride + j + 15];
                  #endif
                  output_filter_channel[i] += (
                                               (
                                                ((a0 + a1) + (a2 + a3))
                  #if STRIDE > 4
                                             +  ((a4 + a5) + (a6 + a7))
                  #endif
                                               )
                  #if STRIDE > 8
                                             + (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15)))
                  #endif
                                              );
               }
               #undef STRIDE

               for (; j < in_channels; ++j)
               {
                  int transposed_stride = i * in_channels;
                  float a0 = temp2[transposed_stride + j + 0];
                  output_filter_channel[i] += (a0);
               }

               output_filter_channel[i] += bias_value;
            }
#else
            /////////////////////////////////////////////////////////////////////////////
            // NOTE(irwin): variant E: SIMD (best one so far)
            //              in_channels are premultiplied by the kernels
            //              then SIMD summed
            /////////////////////////////////////////////////////////////////////////////
            for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
            {

               float *in_channel = input_data_batch + channel_index * array_count;

               float kernel = *index3d( filters, filter_index, channel_index, 0 );

               for (int i = 0; i < array_count; ++i)
               {
                  float *out_channel = temp + in_channels * i;
                  out_channel[channel_index] = in_channel[i] * kernel;
               }
            }

            float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            for (int i = 0; i < array_count; ++i)
            {
               int j = 0;
               int stride = 16;

               __m256 r1 = _mm256_setzero_ps();
               __m256 r2 = _mm256_setzero_ps();
               for (; j < in_channels - (stride - 1); j += stride)
               {
                  int transposed_stride = i * in_channels;
                  __m256 a1 = _mm256_loadu_ps(temp + transposed_stride + j + 0);
                  __m256 a2 = _mm256_loadu_ps(temp + transposed_stride + j + 8);
                  r1 = _mm256_add_ps(r1, a1);
                  r2 = _mm256_add_ps(r2, a2);
               }
               r1 = _mm256_hadd_ps(r1, r2);

               r1 = _mm256_hadd_ps(r1, r1);
               r1 = _mm256_hadd_ps(r1, r1);
               // r = _mm256_hadd_ps(r, r);

               output_filter_channel[i] += ((float *)&r1)[0] + ((float *)&r1)[4];
               // float v;
               // _MM_EXTRACT_FLOAT(v, _mm256_extractf128_ps(r, 0), 0);
               // output_filter_channel[i] += v;


               for (; j < in_channels; ++j)
               {
                  int transposed_stride = i * in_channels;
                  float a0 = temp[transposed_stride + j + 0];
                  output_filter_channel[i] += (a0);
               }

               output_filter_channel[i] += bias_value;
            }
#endif


            endTemporaryMemory(batch_mark);
         }

      }
   }
   else
   {
      for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
      {
         float *input_data_batch = input->data + batch_index * batch_stride_input;
         float *output_data_batch = output->data + batch_index * batch_stride_output;
         for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
         // for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
         {
            // float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            if (biases)
            {
               // float bias_value = biases->data[filter_index];
               // for (int i = 0; i < output_array_count; ++i)
               // {
               //    output_filter_channel[i] = bias_value;
               // }
            }

            // for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
            for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
            {
               float *output_filter_channel = output_data_batch + filter_index * output_array_count;

               float *kernel = index3d( filters, filter_index, channel_index, 0 );

               float *channel = input_data_batch + channel_index * array_count;
               for ( int index = 0; index < output_array_count; ++index )
               {
                  #if 1
                  output_filter_channel[index] += dotproduct_slow( channel + index * hop_length, kernel_size, kernel, kernel_size );
                  #elif 1
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

         for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
         {
            float *output_filter_channel = output_data_batch + filter_index * output_array_count;
            if (biases)
            {
               float bias_value = biases->data[filter_index];
               for (int i = 0; i < output_array_count; ++i)
               {
                  output_filter_channel[i] += bias_value;
               }
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
