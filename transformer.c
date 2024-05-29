#include "tensor.h"

// transformer = TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48, scale=2 * np.sqrt(2))
//
//                                        16                          48
//    self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out, scale=scale)
//       self.QKV = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
//       self.out_proj = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)

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

static void dual_head_attention(MemoryArena *arena, TestTensor *input,
                                 TestTensor *QKV_weights, TestTensor *QKV_biases,
                                 TestTensor *proj_weights, TestTensor *proj_biases,
                                 TestTensor *output )
{
   TracyCZone(dual_head_attention, true);

   Assert( input->ndim == 2 );
   Assert( output->ndim == 2 );

   Assert( QKV_weights->ndim == 2 );
   Assert( QKV_biases->ndim == 1 );

   int in_features = tdim( QKV_weights, -1 );
   const int n_heads = 2;
   int head_length = in_features / n_heads;
   int out_features = tdim( QKV_weights, -2 );
   int seq_length = tdim( input, -2 );

   Assert( in_features == tdim( input, -1 ) );
   Assert( out_features == tdim( QKV_biases, 0 ) );

   Assert( proj_weights->ndim == 2 );
   Assert( proj_biases->ndim == 1 );

   Assert( tdim( proj_weights, 0 ) == in_features );
   Assert( tdim( proj_weights, 1 ) == in_features );
   Assert( tdim( proj_biases, 0 ) == in_features );

   Assert( output->ndim == 2 );
   Assert( tdim( output, -2 ) == seq_length );
   Assert( tdim( output, -1 ) == in_features );


   MemoryArena *debug_arena = arena;
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   TestTensor *QKV_result = tensor_zeros_2d( debug_arena, seq_length, out_features );
   tensor_linear( input, QKV_weights, QKV_biases, QKV_result );

   TestTensor *QKV_result_T = tensor_transpose_last_2d( debug_arena, QKV_result );
   int head_size = seq_length * head_length;

   TestTensor head_ref = {.ndim = 2, .dims = {head_length, seq_length}};
   //int head_ref_dims[2] = {head_length, seq_length};
   //head_ref.dims = head_ref_dims;
   //head_ref.ndim = 2;
   head_ref.size = head_size;
   head_ref.nbytes = head_size * sizeof( float );

   head_ref.data = QKV_result_T->data;
   TestTensor *q1 = tensor_transpose_last_2d( debug_arena, &head_ref );

   head_ref.data += head_size;
   TestTensor *q2 = tensor_transpose_last_2d( debug_arena, &head_ref );


   head_ref.data += head_size;
   TestTensor *k1 = tensor_transpose_last_2d( debug_arena, &head_ref );

   head_ref.data += head_size;
   TestTensor *k2 = tensor_transpose_last_2d( debug_arena, &head_ref );


   head_ref.data += head_size;
   TestTensor *v1 = tensor_copy( debug_arena, &head_ref );


   head_ref.data += head_size;
   TestTensor *v2 = tensor_copy( debug_arena, &head_ref );

   TestTensor *a1 = tensor_zeros_2d( debug_arena, tdim( k1, -2 ), tdim( q1, -2 ) );
   TestTensor *a2 = tensor_zeros_like( debug_arena, a1 );

   tensor_linear( k1, q1, 0, a1 );
   tensor_linear( k2, q2, 0, a2 );



   // NOTE(irwin): 1.0f / sqrtf(head_length);
   //              where head_length is the dimensionality of the head
   //              (this is the sqrt(dk) in the paper Attention Is All You Need
   //              https://arxiv.org/pdf/1706.03762.pdf)
   // NOTE(irwin): this is done for numerical stability
   const float scale = 1.0f / sqrtf((float)head_length);

   tensor_mul_inplace( a1, scale );
   tensor_mul_inplace( a2, scale );

   softmax_inplace_stable( debug_arena, a1 );
   softmax_inplace_stable( debug_arena, a2 );

   // [25, 25] x [8, 25] = [25, 8]
   TestTensor *attn1 = tensor_zeros_2d( debug_arena, tdim( a1, -2 ), tdim( v1, -2 ) );
   TestTensor *attn2 = tensor_zeros_like( debug_arena, attn1 );

   // [25, 8]
   // [25, 8]
   tensor_linear( a1, v1, 0, attn1 );
   tensor_linear( a2, v2, 0, attn2 );

   // [8, 25]
   // [8, 25]
   TestTensor *attn1_t = tensor_transpose_last_2d( debug_arena, attn1 );
   TestTensor *attn2_t = tensor_transpose_last_2d( debug_arena, attn2 );

   // [16, 25]
   // TODO(irwin): tensor_concat routine
   TestTensor *attn12_t = tensor_zeros_2d( debug_arena, tdim( attn1_t, -2 ) * 2, tdim( attn1_t, -1 ) );
   memmove( attn12_t->data, attn1_t->data, attn1_t->nbytes );
   memmove( attn12_t->data + attn1_t->size, attn2_t->data, attn2_t->nbytes );

   // [25, 16]
   TestTensor *attention = tensor_transpose_last_2d( debug_arena, attn12_t );

   // [25, 16] x [16, 16] + [16] = [25, 16]
   tensor_linear( attention, proj_weights, proj_biases, output );

   endTemporaryMemory( mark );
   TracyCZoneEnd(dual_head_attention);
}


// TODO(irwin):
// - [x] batch input support via wrapper
// - [ ] proper batch input support
static void transformer_block( MemoryArena *arena, TestTensor *input,
                               TestTensor *attention_weights, TestTensor *attention_biases,
                               TestTensor *attention_proj_weights, TestTensor *attention_proj_biases,
                               TestTensor *norm1_weights, TestTensor *norm1_biases,
                               TestTensor *linear1_weights, TestTensor *linear1_biases,
                               TestTensor *linear2_weights, TestTensor *linear2_biases,
                               TestTensor *norm2_weights, TestTensor *norm2_biases,
                               TestTensor *output )
{
   TracyCZone(transformer_block, true);

   Assert( input->ndim == 2 );
   Assert( output->ndim == 2 );

   int shape = tdim( input, -2 );

   TemporaryMemory mark = beginTemporaryMemory( arena );

   TestTensor *input_transposed = tensor_transpose_last_2d( arena, input );
   TestTensor *attention_output = tensor_zeros_like( arena, input_transposed );

   dual_head_attention( arena, input_transposed,
                        attention_weights, attention_biases,
                        attention_proj_weights, attention_proj_biases,
                        attention_output );

   tensor_add_inplace_nd( input_transposed, attention_output );

   // TODO(irwin): can zero and reuse attention_output?
   TestTensor *norm1_output = tensor_zeros_like( arena, input_transposed );
   layer_norm( arena, input_transposed, norm1_weights, norm1_biases, norm1_output );

   // NOTE(irwin): tdim(input_transposed, -1) == tdim(input, -2)
   // NOTE(irwin): tdim(norm1_output, -1) == tdim(input_transposed, -1)
   // NOTE(irwin): shape is tdim(input, -2)
   Assert(tdim( norm1_output, -1 ) == shape);
   TestTensor *linear1_output = tensor_zeros_2d( arena, tdim( norm1_output, -2 ), shape );
   tensor_linear( norm1_output, linear1_weights, linear1_biases, linear1_output );
   tensor_relu_inplace( linear1_output );
   TestTensor *linear2_output = tensor_zeros_2d( arena, tdim( linear1_output, -2 ), shape );
   tensor_linear( linear1_output, linear2_weights, linear2_biases, linear2_output );
   tensor_add_inplace_nd( norm1_output, linear2_output );

   TestTensor *norm2_output = tensor_zeros_like( arena, norm1_output );
   layer_norm( arena, norm1_output, norm2_weights, norm2_biases, norm2_output );

   TestTensor *output_copy_source = tensor_transpose_last_2d( arena, norm2_output );
   Assert(output->nbytes == output_copy_source->nbytes);
   memmove( output->data, output_copy_source->data, output->nbytes );

   endTemporaryMemory( mark );
   TracyCZoneEnd(transformer_block);
}

static void transformer_block_batch( MemoryArena *arena, TestTensor *input,
                                     TestTensor *attention_weights, TestTensor *attention_biases,
                                     TestTensor *attention_proj_weights, TestTensor *attention_proj_biases,
                                     TestTensor *norm1_weights, TestTensor *norm1_biases,
                                     TestTensor *linear1_weights, TestTensor *linear1_biases,
                                     TestTensor *linear2_weights, TestTensor *linear2_biases,
                                     TestTensor *norm2_weights, TestTensor *norm2_biases,
                                     TestTensor *output )
{
   TracyCZone(transformer_block_batch, true);

   Assert( input->ndim == 3 );
   Assert( output->ndim == 3 );

   int batch_size = tdim( input, 0 );
   for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      TestTensor input_slice = tensor_index_first_dim( input, batch_index, false );
      TestTensor output_slice = tensor_index_first_dim( output, batch_index, false );

      transformer_block( arena, &input_slice,
                         attention_weights, attention_biases,
                         attention_proj_weights, attention_proj_biases,
                         norm1_weights, norm1_biases,
                         linear1_weights, linear1_biases,
                         linear2_weights, linear2_biases,
                         norm2_weights, norm2_biases,
                         &output_slice );
   }
   TracyCZoneEnd(transformer_block_batch);
}


static void transformer_layer( MemoryArena *arena, TestTensor *input, TransformerLayer_Weights weights, int conv_stride, TestTensor *output )
{
   TracyCZone(transformer_layer, true);


   TemporaryMemory mark = beginTemporaryMemory( arena );

   ConvOutputShape conv_block_out_shape = conv_block_output_shape( input, weights.dw_conv_weights, weights.pw_conv_weights );

   {
      ConvOutputShape output_required_shape = conv_output_shape_shape( conv_block_out_shape, weights.conv_weights, conv_stride );
      // TODO(irwin): verify
      Assert( output_required_shape.batch_size == tdim( output, 0 ) );
      Assert( output_required_shape.channels_out == tdim( output, 1 ) );
      Assert( output_required_shape.sequence_length == tdim( output, 2 ) );
   }

   TestTensor* conv_block_output = tensor_zeros_3d( arena, conv_block_out_shape.batch_size, conv_block_out_shape.channels_out, conv_block_out_shape.sequence_length );

   b32 conv_block_has_proj = (weights.proj_weights != 0 && weights.proj_biases != 0);
   // NOTE(irwin): 1 - ConvBlock
   conv_block( arena, input, conv_block_has_proj,
               weights.dw_conv_weights, weights.dw_conv_biases,
               weights.pw_conv_weights, weights.pw_conv_biases,
               weights.proj_weights, weights.proj_biases,
               conv_block_output );


   TestTensor *transformer_block_output = tensor_zeros_like( arena, conv_block_output );

   // NOTE(irwin): 2 - TransformerBlock
   transformer_block_batch(arena,
                     conv_block_output,
                     weights.attention_weights, weights.attention_biases,
                     weights.attention_proj_weights, weights.attention_proj_biases,
                     weights.norm1_weights, weights.norm1_biases,
                     weights.linear1_weights, weights.linear1_biases,
                     weights.linear2_weights, weights.linear2_biases,
                     weights.norm2_weights, weights.norm2_biases,
                     transformer_block_output);

   // NOTE(irwin): 3 - Conv1d
   int hop_length = conv_stride;
   TestTensor *conv_output = conv_tensor_out ( arena, transformer_block_output, weights.conv_weights, weights.conv_biases, hop_length );

   batch_norm1d( conv_output,
                 weights.batch_norm_running_mean,
                 weights.batch_norm_running_var,
                 weights.batch_norm_weights,
                 weights.batch_norm_biases,
                 output );

   // NOTE(irwin): 4 - ReLU
   tensor_relu_inplace( output );


   endTemporaryMemory( mark );
   TracyCZoneEnd(transformer_layer);
}

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
      {
         int filter_length = tdim( context->weights.forward_basis_buffer, 2 );
         int half_filter_length = filter_length / 2;
         cutoff = half_filter_length + 1;
      }
      // TODO(irwin): dehardcode 64 hop_length
      int stft_out_features_count = compute_stft_output_feature_count( input_one_batch, context->weights.forward_basis_buffer, 64 );
      TestTensor *stft_output = tensor_zeros_3d( debug_arena, tdim( input_one_batch, -2 ), cutoff, stft_out_features_count );

      my_stft( debug_arena, input_one_batch, context->weights.forward_basis_buffer, stft_output );

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
