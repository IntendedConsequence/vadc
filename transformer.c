#include "tensor.h"

// transformer = TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48, scale=2 * np.sqrt(2))
//
//                                        16                          48
//    self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out, scale=scale)
//       self.QKV = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
//       self.out_proj = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)

static void layer_norm( TestTensor *input, TestTensor *weight, TestTensor *bias, TestTensor *output )
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

   MemoryArena *debug_arena = DEBUG_getDebugArena();
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

static void dual_head_attention( TestTensor *input,
                                 TestTensor *QKV_weights, TestTensor *QKV_biases,
                                 TestTensor *proj_weights, TestTensor *proj_biases,
                                 TestTensor *output )
{
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


   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   TestTensor *QKV_result = tensor_zeros_2d( debug_arena, seq_length, out_features );
   tensor_linear( input, QKV_weights, QKV_biases, QKV_result );

   TestTensor *QKV_result_T = tensor_transpose_2d( debug_arena, QKV_result );
   int head_size = seq_length * head_length;

   TestTensor head_ref = {0};
   int head_ref_dims[2] = {head_length, seq_length};
   head_ref.dims = head_ref_dims;
   head_ref.ndim = 2;
   head_ref.size = head_size;
   head_ref.nbytes = head_size * sizeof( float );

   head_ref.data = QKV_result_T->data;
   TestTensor *q1 = tensor_transpose_2d( debug_arena, &head_ref );

   head_ref.data += head_size;
   TestTensor *q2 = tensor_transpose_2d( debug_arena, &head_ref );


   head_ref.data += head_size;
   TestTensor *k1 = tensor_transpose_2d( debug_arena, &head_ref );

   head_ref.data += head_size;
   TestTensor *k2 = tensor_transpose_2d( debug_arena, &head_ref );


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
   TestTensor *attn1_t = tensor_transpose_2d( debug_arena, attn1 );
   TestTensor *attn2_t = tensor_transpose_2d( debug_arena, attn2 );

   // [16, 25]
   // TODO(irwin): tensor_concat routine
   TestTensor *attn12_t = tensor_zeros_2d( debug_arena, tdim( attn1_t, -2 ) * 2, tdim( attn1_t, -1 ) );
   memmove( attn12_t->data, attn1_t->data, attn1_t->nbytes );
   memmove( attn12_t->data + attn1_t->size, attn2_t->data, attn2_t->nbytes );

   // [25, 16]
   TestTensor *attention = tensor_transpose_2d( debug_arena, attn12_t );

   // [25, 16] x [16, 16] + [16] = [25, 16]
   tensor_linear( attention, proj_weights, proj_biases, output );

   endTemporaryMemory( mark );
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
   Assert( input->ndim == 2 );
   Assert( output->ndim == 2 );

   int shape = tdim( input, -2 );

   TemporaryMemory mark = beginTemporaryMemory( arena );

   TestTensor *input_transposed = tensor_transpose_2d( arena, input );
   TestTensor *attention_output = tensor_zeros_like( arena, input_transposed );

   dual_head_attention( input_transposed,
                        attention_weights, attention_biases,
                        attention_proj_weights, attention_proj_biases,
                        attention_output );

   tensor_add_inplace_nd( input_transposed, attention_output );

   // TODO(irwin): can zero and reuse attention_output?
   TestTensor *norm1_output = tensor_zeros_like( arena, input_transposed );
   layer_norm( input_transposed, norm1_weights, norm1_biases, norm1_output );

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
   layer_norm( norm1_output, norm2_weights, norm2_biases, norm2_output );

   TestTensor *output_copy_source = tensor_transpose_2d( arena, norm2_output );
   Assert(output->nbytes == output_copy_source->nbytes);
   memmove( output->data, output_copy_source->data, output->nbytes );

   endTemporaryMemory( mark );
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
   Assert( input->ndim == 3 );
   Assert( output->ndim == 3 );

   int batch_size = tdim( input, 0 );
   for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      TestTensor input_slice = tensor_slice_first_dim( input, batch_index );
      TestTensor output_slice = tensor_slice_first_dim( output, batch_index );

      transformer_block( arena, &input_slice, 
                         attention_weights, attention_biases, 
                         attention_proj_weights, attention_proj_biases, 
                         norm1_weights, norm1_biases, 
                         linear1_weights, linear1_biases, 
                         linear2_weights, linear2_biases, 
                         norm2_weights, norm2_biases, 
                         &output_slice );
   }
}

typedef struct TransformerLayer_Weights TransformerLayer_Weights;
struct TransformerLayer_Weights
{
   // NOTE(irwin): ConvBlock

   TestTensor *dw_conv_weights;
   TestTensor *dw_conv_biases;
   TestTensor *pw_conv_weights;
   TestTensor *pw_conv_biases;
   // NOTE(irwin): optional proj
   TestTensor *proj_weights;
   TestTensor *proj_biases;


   // NOTE(irwin): attention
   TestTensor *attention_weights;
   TestTensor *attention_biases;
   TestTensor *attention_proj_weights;
   TestTensor *attention_proj_biases;

   // NOTE(irwin): transformer rest
   TestTensor *norm1_weights;
   TestTensor *norm1_biases;
   TestTensor *linear1_weights;
   TestTensor *linear1_biases;
   TestTensor *linear2_weights;
   TestTensor *linear2_biases;
   TestTensor *norm2_weights;
   TestTensor *norm2_biases;

   // NOTE(irwin): conv1d
   TestTensor *conv_weights;
   TestTensor *conv_biases;

   // NOTE(irwin): batch norm
   TestTensor *batch_norm_weights;
   TestTensor *batch_norm_biases;
   TestTensor *batch_norm_running_mean;
   TestTensor *batch_norm_running_var;


   // TODO(irwin):
   // - [x] Conv1d (2)
   // - [x] BatchNorm1d (4)
   // - [x] ConvBlock
   //       - [x] Conv1d (2)
   //       - [x] Conv1d (2)
   //       - [x] proj (optional) (2)
};

static void transformer_layer( MemoryArena *arena, TestTensor *input, TransformerLayer_Weights weights, TestTensor *output )
{
   TemporaryMemory mark = beginTemporaryMemory( arena );

   ConvOutputShape conv_block_out_shape = conv_block_output_shape( input, weights.dw_conv_weights, weights.pw_conv_weights );

   {
      ConvOutputShape output_required_shape = conv_output_shape_shape( conv_block_out_shape, weights.conv_weights, 2 );
      // TODO(irwin): verify
      Assert( output_required_shape.batch_size == tdim( output, 0 ) );
      Assert( output_required_shape.channels_out == tdim( output, 1 ) );
      Assert( output_required_shape.sequence_length == tdim( output, 2 ) );
   }

   TestTensor* conv_block_output = tensor_zeros_3d( arena, conv_block_out_shape.batch_size, conv_block_out_shape.channels_out, conv_block_out_shape.sequence_length );
   
   // NOTE(irwin): 1 - ConvBlock
   conv_block( input, 1,
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
   int hop_length = 2;
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
}