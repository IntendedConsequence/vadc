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

static void layer_norm( MemoryArena *arena, TestTensor *input, TestTensor *weight, TestTensor *bias, TestTensor *output );

static void layer_norm_batch( MemoryArena *arena, TestTensor *input, TestTensor *weight, TestTensor *bias, TestTensor *output )
{
   Assert( input->ndim == 3 );
   Assert( output->ndim == 3 );

   int batch_size = tdim( input, 0 );
   for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      TestTensor input_slice = tensor_index_first_dim( input, batch_index, false );
      TestTensor output_slice = tensor_index_first_dim( output, batch_index, false );

      layer_norm( arena, &input_slice, weight, bias, &output_slice );
   }
}

static void layer_norm( MemoryArena *arena, TestTensor *input, TestTensor *weight, TestTensor *bias, TestTensor *output )
{
   const float eps = 1e-5f;

   Assert( input->ndim == 2 );
   Assert( output->ndim == 2 );
   Assert( weight->ndim == 1 );
   Assert( bias->ndim == 1 );

   int batches = tdim( input, 0 );
   int features = tdim( input, 1 );
   Assert( features > 0 );
   float inv_features = 1.0f / features;

   Assert( batches == tdim( output, 0 ) );
   Assert( features == tdim( output, 1 ) );

   Assert( features == tdim( weight, 0 ) );
   Assert( features == tdim( bias, 0 ) );

   MemoryArena *debug_arena = arena;
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   float *mean = pushArray( debug_arena, batches, float );
   float *variance = pushArray( debug_arena, batches, float );

   for ( int batch = 0; batch < batches; ++batch )
   {
      float sum = 0.0f;
      for ( int index = 0; index < features; ++index )
      {
         sum += input->data[batch * features + index];
      }
      mean[batch] = sum * inv_features;
   }

   for ( int batch = 0; batch < batches; ++batch )
   {
      float sum = 0.0f;
      for ( int index = 0; index < features; ++index )
      {
         float diff = input->data[batch * features + index] - mean[batch];
         sum += diff * diff;
      }
      variance[batch] = sum * inv_features;
   }

   for ( int batch = 0; batch < batches; ++batch )
   {
      float std_dev = sqrtf( variance[batch] + eps );
      float std_dev_reciprocal = 1.0f / std_dev;
      float mean_over_std_dev = mean[batch] * std_dev_reciprocal;

      for ( int index = 0; index < features; ++index )
      {
         int array_index = batch * features + index;
         float input_value = input->data[array_index];
#if 0
         float diff = input_value - mean[batch];
         output->data[array_index] = diff * std_dev_reciprocal * weight->data[index] + bias->data[index];
#else
         output->data[array_index] = (input_value * std_dev_reciprocal - mean_over_std_dev) * weight->data[index] + bias->data[index];
#endif
      }
   }

   endTemporaryMemory( mark );
}

/*
def mybatchnorm1d(x, running_mean, running_var, weight, bias, eps=1e-05):
    """simple numpy implementation of batchnorm1d. x is (batches, features, sequence)"""

    x_normalized = (x - running_mean[None, :, None]) / np.sqrt(running_var[None, :, None] + eps)
    x_normalized = x_normalized * weight[None, :, None] + bias[None, :, None]
    return x_normalized
*/

static void batch_norm1d( TestTensor *input,
                          TestTensor *running_mean,
                          TestTensor *running_var,
                          TestTensor *weight,
                          TestTensor *bias,
                          TestTensor *output )
{
   const float eps = 1e-5f;

   Assert( input->ndim == 3 );
   Assert( output->ndim == 3 );
   Assert( running_mean->ndim == 1 );
   Assert( running_var->ndim == 1 );
   Assert( weight->ndim == 1 );
   Assert( bias->ndim == 1 );

   int batches = tdim( input, 0 );
   int features = tdim( input, 1 );
   int sequence = tdim( input, 2 );

   for ( int batch = 0; batch < batches; ++batch )
   {
      for ( int index = 0; index < features; ++index )
      {
         float mean = running_mean->data[index];
         float variance = running_var->data[index];
         float std_dev = sqrtf( variance + eps );

         for ( int sequence_index = 0; sequence_index < sequence; ++sequence_index )
         {
            float value = *index3d( input, batch, index, sequence_index );
            float normalized_value = (value - mean) / std_dev;
            float scaled_value = normalized_value * weight->data[index] + bias->data[index];
            *index3d( output, batch, index, sequence_index ) = scaled_value;
         }
      }
   }
}
