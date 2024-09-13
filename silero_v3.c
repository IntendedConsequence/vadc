
static inline void decoder_tensor ( MemoryArena *arena, TestTensor *input, TestTensor *weights, TestTensor *biases, TestTensor *output );

static void encoder(MemoryArena *arena, TestTensor *input, Encoder_Weights encoder_weights, TestTensor *output)
{
   TracyCZone(encoder, true);

   TemporaryMemory mark = beginTemporaryMemory( arena );

   TestTensor *l1_output = 0;
   {
      ConvOutputShape l1_output_required_shape = shape_for_transformer( input, encoder_weights.l1, encoder_weights.l1_conv_stride );
      l1_output = tensor_zeros_3d( arena, l1_output_required_shape.batch_size, l1_output_required_shape.channels_out, l1_output_required_shape.sequence_length );
   }

   TestTensor *l2_output = 0;
   {
      ConvOutputShape l2_output_required_shape = shape_for_transformer( l1_output, encoder_weights.l2, encoder_weights.l2_conv_stride );
      l2_output = tensor_zeros_3d( arena, l2_output_required_shape.batch_size, l2_output_required_shape.channels_out, l2_output_required_shape.sequence_length );
   }

   TestTensor *l3_output = 0;
   {
      ConvOutputShape l3_output_required_shape = shape_for_transformer( l2_output, encoder_weights.l3, encoder_weights.l3_conv_stride );
      l3_output = tensor_zeros_3d( arena, l3_output_required_shape.batch_size, l3_output_required_shape.channels_out, l3_output_required_shape.sequence_length );
   }

   TestTensor *l4_output = output;
   {
      ConvOutputShape l4_output_required_shape = shape_for_transformer( l3_output, encoder_weights.l4, encoder_weights.l4_conv_stride );

      Assert( output->ndim == 3 );
      Assert( tdim( output, 0 ) == l4_output_required_shape.batch_size );
      Assert( tdim( output, 1 ) == l4_output_required_shape.channels_out );
      Assert( tdim( output, 2 ) == l4_output_required_shape.sequence_length );
   }

   transformer_layer( arena,
                      input,
                      encoder_weights.l1,
                      encoder_weights.l1_conv_stride,
                      l1_output );

   transformer_layer( arena,
                      l1_output,
                      encoder_weights.l2,
                      encoder_weights.l2_conv_stride,
                      l2_output );

   transformer_layer( arena,
                      l2_output,
                      encoder_weights.l3,
                      encoder_weights.l3_conv_stride,
                      l3_output );

   transformer_layer( arena,
                      l3_output,
                      encoder_weights.l4,
                      encoder_weights.l4_conv_stride,
                      l4_output );

   endTemporaryMemory( mark );
   TracyCZoneEnd(encoder);
}

typedef struct One_Batch_Result One_Batch_Result;
struct One_Batch_Result
{
   float unkn;
   float prob;
};
static One_Batch_Result silero_run_one_batch_with_context(MemoryArena *arena, Silero_Context *context, int samples_count, float *samples)
{
   One_Batch_Result result = {0};

   MemoryArena *debug_arena = arena;
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   TestTensor *lstm_input_h = context->state_lstm_h;
   TestTensor *lstm_input_c = context->state_lstm_c;

   TestTensor *lstm_output_h = tensor_zeros_like( debug_arena, lstm_input_h );
   TestTensor *lstm_output_c = tensor_zeros_like( debug_arena, lstm_input_h );

   TestTensor *input_one_batch = tensor_zeros_2d( debug_arena, 1, samples_count );
   memmove(input_one_batch->data, samples, sizeof(float) * samples_count);
   TestTensor *output = tensor_zeros_3d( debug_arena, 1, 2, 1 );

   {
      TemporaryMemory batch_mark = beginTemporaryMemory( debug_arena );

      int cutoff;
      int half_filter_length;
      {
         int filter_length = tdim( context->weights.forward_basis_buffer, 2 );
         half_filter_length = filter_length / 2;
         cutoff = half_filter_length + 1;
      }
      // TODO(irwin): dehardcode 64 hop_length
      int stft_out_features_count = compute_stft_output_feature_count( input_one_batch, context->weights.forward_basis_buffer, 64, half_filter_length );
      TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input_one_batch, -2 ), cutoff, stft_out_features_count );

      my_stft( debug_arena, input_one_batch, context->weights.forward_basis_buffer, stft_output, 64, 128 );

      TestTensor *normalization_output = tensor_copy( debug_arena, stft_output );

      adaptive_audio_normalization_inplace( debug_arena, normalization_output );

      ConvOutputShape l4_output_required_shape = shape_for_encoder( normalization_output, context->weights.encoder_weights );
      TestTensor *l4_output = tensor_zeros_3d( debug_arena, l4_output_required_shape.batch_size, l4_output_required_shape.channels_out, l4_output_required_shape.sequence_length );

      encoder( debug_arena, normalization_output, context->weights.encoder_weights, l4_output );

      TestTensor *l4_output_t = tensor_transpose_last_2d( debug_arena, l4_output );

      int batches = tdim( l4_output_t, -3 );
      int seq_length = tdim( l4_output_t, -2 );
      int input_size = tdim( l4_output_t, -1 );
      int layer_count = tdim( context->weights.lstm_weights, 0 );
      int hidden_size = tdim( context->weights.lstm_weights, -1 ) / 2;
      Assert( hidden_size == input_size );
      Assert( hidden_size == tdim( context->weights.lstm_biases, -1 ) / 4 );
      int batch_stride = seq_length * input_size;
      int lstm_output_size = batch_stride * batches + (input_size * layer_count * 2);

      int hc_size = input_size * layer_count;
      Assert( hc_size == lstm_input_h->size );

      float *lstm_output = pushArray( debug_arena, lstm_output_size, float );
      //float *lstm_output = pushArray( debug_arena, batches * seq_length * input_size, float );

      lstm_seq( arena, l4_output_t->data,
                seq_length * batches,
                input_size,
                lstm_input_h->data,
                lstm_input_c->data,
                context->weights.lstm_weights->data,
                context->weights.lstm_biases->data,
                lstm_output, 2
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

      int decoder_output_size = batches * tdim( context->weights.decoder_weights, 0 );

      int decoder_results = tdim( context->weights.decoder_weights, 0 );
      TestTensor *output_decoder = tensor_zeros_3d( debug_arena, 1, decoder_results, 1 );
      Assert( decoder_output_size == output_decoder->size );

      decoder_tensor(debug_arena, lstm_output_tensor_t, context->weights.decoder_weights, context->weights.decoder_biases, output_decoder );

      float diarization_maybe = output_decoder->data[0];
      float speech_probability = output_decoder->data[1];

      endTemporaryMemory( batch_mark );

      output->data[0] = diarization_maybe;
      output->data[1] = speech_probability;

      memmove( lstm_input_h->data, lstm_output_h->data, lstm_input_h->nbytes );
      memmove( lstm_input_c->data, lstm_output_c->data, lstm_input_c->nbytes );
   }

   result.unkn = output->data[0];
   result.prob = output->data[1];

   endTemporaryMemory( mark );

   return result;
}


// TODO(irwin): simplify according to:
// inputx = torch.randn(1, 64, 7)
// weight = torch.randn(2, 64, 1)
// bias = torch.randn(weight.shape[0])

// torch_decoder = torch.nn.functional.conv1d(inputx.relu(), weight, bias).mean(2, keepdim=True).sigmoid()
// ttt=inputx.relu().sum(-1).squeeze() @ (weight / 7) // weight / inputx.shape[-1]
// np.allclose(sigmoid(ttt.squeeze() + bias).numpy(), torch_decoder.reshape(2).numpy()) // TRUE!

// return self.conv1d(x.relu()).mean(axis=2, keepdim=True).sigmoid()
// input [N, 64, 7]
// weight [2, 64, 1]
// bias [2]
static inline void decoder ( MemoryArena *arena, float *input, int *input_dims, int input_ndims, float *weights, int *weights_dims, int weights_ndims, float *biases, int *biases_dims, int biases_ndims, float *output, int *output_dims, int output_ndims )
{
   VAR_UNUSED( biases_dims );
   VAR_UNUSED( output_dims );
   VAR_UNUSED( biases_ndims );
   VAR_UNUSED( output_ndims );
   VAR_UNUSED( weights_ndims );

   MemoryArena *debug_arena = arena;

   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   Assert( input_ndims == 3 );
   Assert( weights_ndims == 3 );
   Assert( biases_ndims == 1 );
   Assert( output_ndims == 3 );

   {
      int input_count = 1;
      for ( int i = 0; i < input_ndims; ++i )
      {
         input_count *= input_dims[i];
      }

      int input_size = input_count * sizeof( float );

      float *relu_result = pushArray( debug_arena, input_count, float );
      memcpy( relu_result, input, input_size ); // TODO(irwin): memmove?
      relu_inplace( relu_result, input_count );

      int batch_count = input_dims[0];

      // [N, 2, 7]
      int convolve_result_count = 1;
      convolve_result_count *= batch_count;
      convolve_result_count *= weights_dims[0];
      convolve_result_count *= input_dims[2];

      // if (convolve_result_count != 14)
      // {
      //     return 0;
      // }


      float *convolve_result = pushArray( debug_arena, convolve_result_count, float );
      convolve_mc_mf_batch_bias( batch_count, relu_result, input_dims[1], input_dims[2], weights, weights_dims[0], convolve_result, biases );

      // return __LINE__;
      // [N, 2, 1]
      int mean_output_count = 1;
      mean_output_count *= batch_count;
      mean_output_count *= weights_dims[0];

      // float *mean_result = vadc_malloc(mean_output_count * sizeof(float));
      float *mean_result = output;

      int input_offset = 0;
      int output_offset = 0;
      for ( int b = 0; b < batch_count; ++b )
      {
         for ( int f = 0; f < weights_dims[0]; ++f )
         {
            float mean_value = mean( convolve_result + input_offset, input_dims[2] );

            // TODO(irwin): sigmoid
            mean_result[output_offset++] = 1.0f / (1.0f + expf( -mean_value ));
            input_offset += input_dims[2];
         }
      }
   }

   endTemporaryMemory( mark );
}

static inline void decoder_tensor ( MemoryArena *arena, TestTensor *input, TestTensor *weights, TestTensor *biases, TestTensor *output )
{
   decoder( arena, input->data, input->dims, input->ndim,
                   weights->data, weights->dims, weights->ndim,
                   biases->data, biases->dims, biases->ndim,
                   output->data, output->dims, output->ndim );
}

