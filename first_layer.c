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

static void conv_tensor ( TestTensor *input, TestTensor *filters, TestTensor *biases, int hop_length, TestTensor *output )
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

   for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      float *input_data_batch = input->data + batch_index * batch_stride_input;
      float *output_data_batch = output->data + batch_index * batch_stride_output;
      for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
      {
         float *output_filter_channel = output_data_batch + filter_index * output_array_count;
         if (biases)
         {
            for (int i = 0; i < output_array_count; ++i)
            {
               output_filter_channel[i] = biases->data[filter_index];
            }
         }

         for ( int channel_index = 0; channel_index < in_channels; ++channel_index )
         {
            float *kernel = index3d( filters, filter_index, channel_index, 0 );

            float *channel = input_data_batch + channel_index * array_count;
            for ( int index = 0; index < output_array_count; ++index )
            {
               output_filter_channel[index] += dotproduct( channel + index * hop_length, kernel_size, kernel, kernel_size );
            }
         }
      }
   }

   TracyCZoneEnd(conv_tensor);
}


static inline TestTensor *conv_tensor_out ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *biases, int hop_length )
{
   TracyCZone(conv_tensor_out, true);

   TestTensor *output = tensor_zeros_for_conv( arena, input, filters, hop_length );
   conv_tensor( input, filters, biases, hop_length, output );

   TracyCZoneEnd(conv_tensor_out);
   return output;
}

static void pw_conv_tensor ( TestTensor *input, TestTensor *filters, TestTensor *biases, TestTensor *output )
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

static TestTensor *tensor_reflect_pad_last_dim( MemoryArena *arena, TestTensor *input, int padding )
{
   int last_dim_index = tdimindex( input, -1 );
   int last_dim = input->dims[last_dim_index];
   int last_dim_padded = last_dim + 2 * padding;

   int *new_dims = pushArray( arena, input->ndim, int );
   for ( int i = 0; i < input->ndim; ++i )
   {
      new_dims[i] = input->dims[i];
   }
   new_dims[last_dim_index] = last_dim_padded;

   int rows = input->size / last_dim;
   TestTensor *new_tensor = tensor_zeros( arena, input->ndim, new_dims );

   for (int i = 0; i < rows; ++i)
   {
      float *input_row = input->data + i * last_dim;
      float *output_row_start = new_tensor->data + i * last_dim_padded;

      float *output_row_unpadded = output_row_start + padding;
      memcpy( output_row_unpadded, input_row, last_dim * sizeof(float) );

      float *output_row_pad_left_cursor = output_row_start;
      float *output_row_pad_right_cursor = output_row_start + last_dim_padded - padding;

      float *input_row_reflect_left_cursor = input_row + padding;
      float *input_row_reflect_right_cursor = input_row + last_dim - 2;

      for (int j = 0; j < padding; ++j)
      {
         *output_row_pad_left_cursor++ = *input_row_reflect_left_cursor--;
         *output_row_pad_right_cursor++ = *input_row_reflect_right_cursor--;
      }
   }

   return new_tensor;
}

static void adaptive_audio_normalization_inplace(MemoryArena *arena, TestTensor *input)
{
   TracyCZone(adaptive_audio_normalization_inplace, true);

   static float filter[7] = {
      0.03663284704089164733887f,
      0.11128076165914535522461f,
      0.21674531698226928710938f,
      0.27068215608596801757812f,
      0.21674531698226928710938f,
      0.11128076165914535522461f,
      0.03663284704089164733887f
   };

   static TestTensor filter_tensor = {
      .ndim = 3,
      .dims = {1, 1, 7},
      .data = filter,
      .size = 7,
      .nbytes = 7 * sizeof(float),
      .name = NULL,
   };

   const int to_pad = 3; // (kernel size - 1) / 2
   const float million = (float)(1024 * 1024); // 1048576

   TemporaryMemory mark = beginTemporaryMemory(arena);
   // 1) tensor for mean [:, :, tdim(input, -1)]
   // 2) tensor for pad-reflected-mean [:, :, tdim(mean, -1) + to_pad * 2]
   // 3) tensor for conv1d output [:, :, tdim(mean, -1)] // can reuse mean from 1)

   TestTensor input_unsqueezed = *input;
   if (input->ndim == 2)
   {
      input_unsqueezed = tensor_unsqueeze(input, 0);
   }

   TestTensor *mean = tensor_zeros_3d(arena, input_unsqueezed.dims[0], 1, input_unsqueezed.dims[2]);

   for (int i = 0; i < input_unsqueezed.size; ++i)
   {
      float spect = input_unsqueezed.data[i];
      float spect_rescaled = spect * million;
      float spect_log = log1pf(spect_rescaled);
      input_unsqueezed.data[i] = spect_log;
   }

   int channel_count = input_unsqueezed.dims[1];
   int batch_count = input_unsqueezed.dims[0];
   for (int batch_index = 0; batch_index < batch_count; ++batch_index)
   {
      for (int bin_index = 0; bin_index < input_unsqueezed.dims[2]; ++bin_index)
      {
         float bin_sum = 0.0f;
         for (int channel_index = 0; channel_index < channel_count; ++channel_index)
         {
            float value = *index3d(&input_unsqueezed, batch_index, channel_index, bin_index);
            bin_sum += value;
         }
         float bin_mean = bin_sum / channel_count;
         *index3d(mean, batch_index, 0, bin_index) = bin_mean;
      }
   }

   TestTensor *mean_padded = tensor_reflect_pad_last_dim(arena, mean, to_pad);
   TestTensor *conv1d_output = tensor_zeros_like(arena, mean);
   conv_tensor(mean_padded, &filter_tensor, NULL, 1, conv1d_output);

   TestTensor *mean_mean = tensor_zeros_3d(arena, batch_count, 1, 1);

   for (int batch_index = 0; batch_index < batch_count; ++batch_index)
   {
      float mean_sum = 0.0f;
      int bin_count = tdim(conv1d_output, -1);
      for (int i = 0; i < bin_count; ++i)
      {
         float value = *index3d(conv1d_output, batch_index, 0, i);
         mean_sum += value;
      }

      *index3d(mean_mean, batch_index, 0, 0) = mean_sum / bin_count;
   }

   for (int batch_index = 0; batch_index < batch_count; ++batch_index)
   {
      float mean_value = *index3d(mean_mean, batch_index, 0, 0);
      for (int channel_index = 0; channel_index < channel_count; ++channel_index)
      {
         float *channel = index3d(&input_unsqueezed, batch_index, channel_index, 0);
         for (int bin_index = 0; bin_index < input_unsqueezed.dims[2]; ++bin_index)
         {
            float value_adjusted = channel[bin_index] - mean_value;
            channel[bin_index] = value_adjusted;
         }
      }
   }

   endTemporaryMemory(mark);
   /*
   class AdaptiveAudioNormalization(torch.nn.Module):
    filter_: torch.Tensor
    to_pad: int

    def __init__(self):
        super().__init__()

        self.to_pad = 3

        self.register_buffer("filter_", torch.zeros((1, 1, 7)))

    def forward(self, spect: torch.Tensor) -> torch.Tensor:
        spect = torch.log1p(spect * 1048576)
        if len(spect.shape) == 2:
            spect = spect[None, :, :]
        mean = spect.mean(dim=1, keepdim=True)
        mean = simple_pad(mean, self.to_pad)
        mean = torch.conv1d(mean, self.filter_)
        mean_mean = mean.mean(dim=-1, keepdim=True)
        spect = spect.add(-mean_mean)
        return spect
   */

   TracyCZoneEnd(adaptive_audio_normalization_inplace);
}

static inline int compute_stft_output_feature_count( TestTensor *input, TestTensor *filters, int hop_length )
{
   int filter_length = tdim( filters, 2 );
   int padding = filter_length / 2;
   int features_count = (tdim( input, -1 ) + padding * 2 - filter_length) / hop_length + 1; // padded (128 + 1536 + 128) - filter first (256) / hop length (64) + filter first (1)
   
   return features_count;
}

static void my_stft ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *output )
{
   TracyCZone(my_stft, true);

   Assert(filters->ndim == 3);
   Assert(output->ndim == filters->ndim);

   Assert(tdim(filters, 0) == 258);
   Assert(tdim(filters, 1) == 1);
   Assert(tdim(filters, 2) == 256);

   int filter_length = tdim(filters, 2);
   int padding = filter_length / 2;
   Assert(padding * 2 == filter_length);
   int half_filter_length = filter_length / 2;
   int cutoff = half_filter_length + 1;
   int hop_length = 64;
   //int features_count = (tdim(input, -1) + padding * 2 - filter_length) / hop_length + 1; // padded (128 + 1536 + 128) - filter first (256) / hop length (64) + filter first (1)
   
   int features_count = compute_stft_output_feature_count( input, filters, hop_length );

   Assert(tdim(output, 0) == tdim(input, 0));
   Assert(tdim(output, 1) == cutoff);
   Assert(tdim(output, 2) == features_count);


   TemporaryMemory mark = beginTemporaryMemory( arena );

   // int mock_biases_dims[1] = { filters->dims[0] };
   // TestTensor *biases = tensor_zeros( arena, ArrayCount(mock_biases_dims), mock_biases_dims );

   TestTensor input_3d = {0};
   if (input->ndim == 2)
   {
      input_3d = tensor_unsqueeze( input, 1 );
   }
   else
   {
      input_3d = *input;
   }

   TestTensor *input_padded = tensor_reflect_pad_last_dim( arena, &input_3d, padding );

   int output_ndim = 3;
   int output_dims[3] = {0};
   output_dims[0] = tdim( &input_3d, 0); // 1 (+, if batched)
   output_dims[1] = tdim(filters, 0); // 258
   output_dims[2] = features_count; // 25

   TestTensor *conv_output = tensor_zeros(arena, output_ndim, output_dims);
   conv_tensor_stride64_nobias( arena, input_padded, filters, conv_output );

   // [1, 258, 25]
   int batches = tdim(output, 0);
   for (int batch_index = 0; batch_index < batches; ++batch_index )
   {
      int batch_stride = conv_output->size / batches; // 6450 (258*25)
      int real_offset = 0;
      int imag_offset = batch_stride / 2; // 6450 (258*25) / 2 = 3225

      int real_index = batch_index * batch_stride + real_offset;
      int imag_index = batch_index * batch_stride + imag_offset;

      int output_batch_stride = cutoff * features_count; // or output->size / batches
      for (int i = 0; i < output_batch_stride; ++i)
      {
         float real_part = conv_output->data[real_index + i];
         float imag_part = conv_output->data[imag_index + i];
         float magnitude = sqrtf(real_part * real_part + imag_part * imag_part);

         output->data[batch_index * output_batch_stride + i] = magnitude;
      }
   }

   /*
   real_part = forward_transform[:, :cutoff, :]
   imag_part = forward_transform[:, cutoff:, :]

   magnitude = Tensor.sqrt(real_part ** 2 + imag_part ** 2)
   */

   endTemporaryMemory( mark );
   TracyCZoneEnd(my_stft);
}



static void conv_block( TestTensor *input, b32 has_out_proj,
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

   MemoryArena *debug_arena = DEBUG_getDebugArena();
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
