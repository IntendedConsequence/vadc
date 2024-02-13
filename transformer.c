#include "tensor.h"

// transformer = TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48, scale=2 * np.sqrt(2))
//
//                                        16                          48
//    self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out, scale=scale)
//       self.QKV = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
//       self.out_proj = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)

static void tensor_add_inplace( TestTensor *lhs, TestTensor *rhs )
{
   Assert( lhs->size == rhs->size );
   Assert( lhs->ndim == rhs->ndim );
   Assert( 0 == memcmp( lhs->dims, rhs->dims, lhs->ndim ) );

   add_arrays_inplace( lhs->data, lhs->size, rhs->data );
}

TestTensor *tensor_transpose_2d( MemoryArena *arena, TestTensor *source )
{
   TestTensor *output = tensor_zeros_like( arena, source );
   float *data = output->data;
   for ( int x = 0; x < source->dims[1]; ++x )
   {
      for ( int y = 0; y < source->dims[0]; ++y )
      {
         float value = *index2d( source, y, x );
         *data++ = value;
      }
   }

   Assert( data - output->data == output->size );

   output->dims[0] = source->dims[1];
   output->dims[1] = source->dims[0];

   return output;
}

static void tensor_linear( TestTensor *input,
                           TestTensor *weights, TestTensor *biases,
                           TestTensor *output )
{
   Assert( input->ndim == 2 );
   int mata_rows = input->dims[input->ndim - 2];
   int mata_cols = input->dims[input->ndim - 1];

   Assert( weights->ndim == 2 );
   int matb_rows = weights->dims[weights->ndim - 2];
   int matb_cols = weights->dims[weights->ndim - 1];
   mymatmul( input->data, mata_rows, mata_cols, weights->data, matb_rows, matb_cols, output->data );

   Assert( output->ndim == 2 );
   Assert( output->dims[0] == mata_rows && output->dims[1] == matb_rows );
   if ( biases )
   {
      Assert( matb_rows == output->dims[1] && matb_rows == biases->size );
      for ( int i = 0; i < mata_rows; ++i )
      {
         add_arrays_inplace( index2d( output, i, 0 ), biases->size, biases->data );
      }
   }
}

static inline int tdimindex( TestTensor *tensor, int idx )
{
   Assert( tensor->ndim > 0 );
   Assert( -tensor->ndim <= idx && idx < tensor->ndim );
   // ndim idx dim
   // 1     0   0
   // 1    -1   0
   // 2     0   0
   // 2     1   1
   // 2    -1   1
   // 2    -2   0
   // 3     0   0
   // 3     1   1
   // 3     2   2
   // 3    -1   2
   // 3    -2   1
   // 3    -3   0
   return idx < 0 ? tensor->ndim + idx : idx;
}

static inline int tdim( TestTensor *tensor, int idx )
{
   return tensor->dims[tdimindex( tensor, idx )];
}


static inline void softmax_inplace_stable( MemoryArena *arena, TestTensor *input )
{
   TemporaryMemory mark = beginTemporaryMemory( arena );

   TestTensor *exped = tensor_zeros_like( arena, input );
   int stride = tdim( input, -1 );
   int batch_size = input->size / stride;
   for ( int batch_index = 0; batch_index < batch_size; ++batch_index )
   {
      float max_value = input->data[batch_index * stride];
      float sumexp = 0.0f;
      for ( int i = 0; i < stride; ++i )
      {
         float value = input->data[batch_index * stride + i];
         if ( value > max_value )
         {
            max_value = value;
         }
      }
      for ( int i = 0; i < stride; ++i )
      {
         float value = input->data[batch_index * stride + i];
         float e_value = expf( value - max_value );
         exped->data[batch_index * stride + i] = e_value;
         sumexp += e_value;
      }
      float sumexp_inv = 1.0f / sumexp;
      for ( int i = 0; i < stride; ++i )
      {
         input->data[batch_index * stride + i] = exped->data[batch_index * stride + i] * sumexp_inv;
      }
   }
   endTemporaryMemory( mark );
}

static inline void tensor_mul_inplace( TestTensor *input, float value )
{
   for ( int i = 0; i < input->size; ++i )
   {
      input->data[i] *= value;
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



   // NOTE(irwin): 1.0f / (2.0f * sqrtf(2.0f));
   const float scale = 0.3535533905932737622f;
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