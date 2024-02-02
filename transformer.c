#include "tensor.h"

// transformer = TransformerLayer(shape=16, att_qkv_in=16, att_qkv_out=48, scale=2 * np.sqrt(2))
//
//                                        16                          48
//    self.attention = MultiHeadAttention(qkv_in_features=att_qkv_in, qkv_out_features=att_qkv_out, scale=scale)
//       self.QKV = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_out_features)
//       self.out_proj = torch.nn.Linear(in_features=qkv_in_features, out_features=qkv_in_features)

static void tensor_add_inplace(TestTensor *lhs, TestTensor *rhs)
{
   Assert(lhs->size == rhs->size);
   Assert(lhs->ndim == rhs->ndim);
   Assert(0 == memcmp(lhs->dims, rhs->dims, lhs->ndim));

   add_arrays_inplace(lhs->data, lhs->size, rhs->data);
}

TestTensor *tensor_transpose_2d(MemoryArena *arena, TestTensor *source)
{
   TestTensor *output = tensor_zeros_like(arena, source);
   float *data = output->data;
   for (int x = 0; x < source->dims[1]; ++x)
   {
      for (int y = 0; y < source->dims[0]; ++y)
      {
         float value = *index2d(source, y, x);
         *data++ = value;
      }
   }

   Assert(data - output->data == output->size);

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
   Assert( matb_rows == output->dims[1] && matb_rows == biases->size );
   for ( int i = 0; i < matb_rows; ++i )
   {
      add_arrays_inplace(index2d(output, i, 0), biases->size, biases->data);
   }
}

static void multi_head_attention( TestTensor *input,
                                  TestTensor *QKV_weights, TestTensor *QKV_biases,
                                  TestTensor *proj_weights, TestTensor *proj_biases,
                                  TestTensor *output )
{
   VAR_UNUSED(input);
   VAR_UNUSED(QKV_weights);
   VAR_UNUSED(QKV_biases);
   VAR_UNUSED(proj_weights);
   VAR_UNUSED(proj_biases);
   VAR_UNUSED(output);
}