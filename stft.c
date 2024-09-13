static inline int compute_stft_output_feature_count_lr( TestTensor *input, TestTensor *filters, int hop_length, int pad_left, int pad_right )
{
   int filter_length = tdim( filters, 2 );
   // int padding = filter_length / 2;
   int features_count = (tdim( input, -1 ) + (pad_left + pad_right) - filter_length) / hop_length + 1; // padded (128 + 1536 + 128) - filter first (256) / hop length (64) + filter first (1)

   return features_count;
}

static inline int compute_stft_output_feature_count( TestTensor *input, TestTensor *filters, int hop_length, int padding )
{
   return compute_stft_output_feature_count_lr( input, filters, hop_length, padding, padding );
}

static void my_stft_ ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *output, int hop_length, int pad_left, int pad_right )
{
   TracyCZone(my_stft, true);

   Assert(filters->ndim == 3);
   Assert(output->ndim == filters->ndim);

   Assert(tdim(filters, 0) == 258);
   Assert(tdim(filters, 1) == 1);
   Assert(tdim(filters, 2) == 256);

   int filter_length = tdim(filters, 2);
   // int padding = filter_length / 2;
   // Assert(padding * 2 == filter_length);
   int half_filter_length = filter_length / 2;
   int cutoff = half_filter_length + 1;
   // int hop_length = 64;

   int features_count = compute_stft_output_feature_count_lr( input, filters, hop_length, pad_left, pad_right );

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

   TestTensor *input_padded = tensor_reflect_pad_last_dim_lr( arena, &input_3d, pad_left, pad_right );

   int output_ndim = 3;
   int output_dims[3] = {0};
   output_dims[0] = tdim( &input_3d, 0); // 1 (+, if batched)
   output_dims[1] = tdim(filters, 0); // 258
   output_dims[2] = features_count; // 25

   TestTensor *conv_output = tensor_zeros(arena, output_ndim, output_dims);
#if VADC_SLOW
   conv_tensor_stride64_nobias( arena, input_padded, filters, conv_output );
#else // VADC_SLOW
   {
      int batch_size = tdim(input_padded, 0);
      int array_count = tdim(input_padded, -1);

      int out_channels = filters->dims[0];
      int filter_count = out_channels;
      int kernel_size = filters->dims[2];
      int output_array_count = 1 + (array_count - kernel_size) / hop_length;

      int batch_stride_input = input_padded->size / batch_size;
      int batch_stride_output = conv_output->size / batch_size;

      // float *to_sum = pushArray(arena, kernel_size, float);

#if 1
      for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
      {
         float *input_data_batch = input_padded->data + batch_index * batch_stride_input;
         float *output_data_batch = conv_output->data + batch_index * batch_stride_output;
         for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
         {
            float *output_filter_channel = output_data_batch + filter_index * output_array_count;

            float *kernel = index3d( filters, filter_index, 0, 0 );
#else
      for ( int filter_index = 0; filter_index < filter_count; ++filter_index )
      {
         float *kernel = index3d( filters, filter_index, 0, 0 );

         for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
         {
            float *input_data_batch = input_padded->data + batch_index * batch_stride_input;
            float *output_data_batch = conv_output->data + batch_index * batch_stride_output;
            float *output_filter_channel = output_data_batch + filter_index * output_array_count;

#endif
            float *channel = input_data_batch;
            for ( int index = 0; index < output_array_count; ++index )
            {
               float *channel_sub = channel + index * hop_length;

               __m256 r[4];

               r[0] = _mm256_setzero_ps();
               r[1] = _mm256_setzero_ps();
               r[2] = _mm256_setzero_ps();
               r[3] = _mm256_setzero_ps();

               for ( int i = 0; i < kernel_size; i+= 64 )
               {
                  __m256 a1 = _mm256_loadu_ps(channel_sub + i);
                  __m256 b1 = _mm256_loadu_ps(kernel      + i);

                  __m256 a2 = _mm256_loadu_ps(channel_sub + i + 8);
                  __m256 b2 = _mm256_loadu_ps(kernel      + i + 8);

                  __m256 a3 = _mm256_loadu_ps(channel_sub + i + 16);
                  __m256 b3 = _mm256_loadu_ps(kernel      + i + 16);

                  __m256 a4 = _mm256_loadu_ps(channel_sub + i + 24);
                  __m256 b4 = _mm256_loadu_ps(kernel      + i + 24);

                  __m256 a5 = _mm256_loadu_ps(channel_sub + i + 32);
                  __m256 b5 = _mm256_loadu_ps(kernel      + i + 32);

                  __m256 a6 = _mm256_loadu_ps(channel_sub + i + 40);
                  __m256 b6 = _mm256_loadu_ps(kernel      + i + 40);

                  __m256 a7 = _mm256_loadu_ps(channel_sub + i + 48);
                  __m256 b7 = _mm256_loadu_ps(kernel      + i + 48);

                  __m256 a8 = _mm256_loadu_ps(channel_sub + i + 56);
                  __m256 b8 = _mm256_loadu_ps(kernel      + i + 56);

                  __m256 ab1 = _mm256_mul_ps(a1, b1);
                  __m256 ab2 = _mm256_mul_ps(a2, b2);
                  __m256 ab12 = _mm256_add_ps(ab1, ab2);

                  __m256 ab3 = _mm256_mul_ps(a3, b3);
                  __m256 ab4 = _mm256_mul_ps(a4, b4);
                  __m256 ab34 = _mm256_add_ps(ab3, ab4);

                  __m256 ab5 = _mm256_mul_ps(a5, b5);
                  __m256 ab6 = _mm256_mul_ps(a6, b6);
                  __m256 ab56 = _mm256_add_ps(ab5, ab6);

                  __m256 ab7 = _mm256_mul_ps(a7, b7);
                  __m256 ab8 = _mm256_mul_ps(a8, b8);
                  __m256 ab78 = _mm256_add_ps(ab7, ab8);

                  __m256 ab1234 = _mm256_add_ps(ab12, ab34);
                  __m256 ab5678 = _mm256_add_ps(ab56, ab78);

                  r[i / 64] = _mm256_add_ps(ab1234, ab5678);

                  // r = _mm256_add_ps(r, ab);
               }

               __m256 r01 = _mm256_add_ps(r[0], r[1]);
               __m256 r23 = _mm256_add_ps(r[2], r[3]);
               __m256 r0123 = _mm256_add_ps(r01, r23);

               #if 0
               output_filter_channel[index] = ((float *)&r0123)[0] + ((float *)&r0123)[1] +
                                              ((float *)&r0123)[2] + ((float *)&r0123)[3] +
                                              ((float *)&r0123)[4] + ((float *)&r0123)[5] +
                                              ((float *)&r0123)[6] + ((float *)&r0123)[7];
               #else

               float s01 = ((float *)&r0123)[0] + ((float *)&r0123)[1];
               float s23 = ((float *)&r0123)[2] + ((float *)&r0123)[3];
               float s45 = ((float *)&r0123)[4] + ((float *)&r0123)[5];
               float s67 = ((float *)&r0123)[6] + ((float *)&r0123)[7];

               float s0123 = s01 + s23;
               float s4567 = s45 + s67;

               output_filter_channel[index] = s0123 + s4567;

               #endif

            }
         }
      }
   }
#endif // VADC_SLOW
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

static void my_stft ( MemoryArena *arena, TestTensor *input, TestTensor *filters, TestTensor *output, int hop_length, int padding )
{
   my_stft_ ( arena, input, filters, output, hop_length, padding, padding );
}
