#pragma once
#include "maths.h"
#include "memory.h"

typedef struct TestTensor TestTensor;

struct TestTensor
{
   int ndim;
   int dims[8];
   int size;
   int nbytes;
   const char *name;
   float *data;
};

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
};

typedef struct Encoder_Weights Encoder_Weights;
struct Encoder_Weights
{
   TransformerLayer_Weights l1;
   TransformerLayer_Weights l2;
   TransformerLayer_Weights l3;
   TransformerLayer_Weights l4;

   int l1_conv_stride;
   int l2_conv_stride;
   int l3_conv_stride;
   int l4_conv_stride;
};

static inline int tdimindex( TestTensor *tensor, int idx );
static inline int tdim( TestTensor *tensor, int idx );

// static_assert(sizeof(TestTensor) == 64, "Wrong size");

// TODO(irwin):
// - [ ] use where applicable
static inline float *index2d( TestTensor *tensor, int index0, int index1 )
{
   Assert( tensor->ndim == 2 );
   Assert (index0 < tdim(tensor, 0));
   Assert (index1 < tdim(tensor, 1));

   int dim0_stride = tdim(tensor, -1);
   return tensor->data + index0 * dim0_stride + index1;
}

static inline float *index3d( TestTensor *tensor, int index0, int index1, int index2 )
{
   Assert( tensor->ndim == 3 );
   Assert (index0 < tdim(tensor, 0));
   Assert (index1 < tdim(tensor, 1));
   Assert (index2 < tdim(tensor, 2));

   int dim0_stride = tensor->size / tdim(tensor, 0);
   int dim1_stride = tdim(tensor, -1);
   return tensor->data + index0 * dim0_stride + index1 * dim1_stride + index2;
}

static inline b32 tensor_is_valid( TestTensor *tensor )
{
   return (!!tensor->data) & (!!tensor->nbytes) & (!!tensor->size) & (!!tensor->ndim) & (!!tensor->dims);
}

static inline TestTensor tensor_unsqueeze( TestTensor *tensor, int dim )
{
   Assert( tensor_is_valid( tensor ) );

   dim = tdimindex( tensor, dim );

   TestTensor result = {0};

   result.ndim = tensor->ndim + 1;
   for (int i = 0; i < dim; ++i)
   {
      result.dims[i] = tensor->dims[i];
   }
   result.dims[dim] = 1;
   for (int i = dim + 1; i < result.ndim; ++i)
   {
      result.dims[i] = tensor->dims[i-1];
   }

   result.size = tensor->size;
   result.nbytes = tensor->nbytes;
   result.data = tensor->data;
   result.name = tensor->name;

   return result;
}

static inline TestTensor tensor_squeeze( TestTensor *tensor, int dim )
{
   Assert( tensor_is_valid( tensor ) );

   dim = tdimindex( tensor, dim );
   Assert( dim < tensor->ndim );
   Assert( tensor->ndim > 1 );
   Assert( tdim(tensor, dim) == 1 );

   TestTensor result = {0};

   result.ndim = tensor->ndim - 1;
   for (int i = 0; i < dim; ++i)
   {
      result.dims[i] = tensor->dims[i];
   }
   for (int i = dim; i < result.ndim; ++i)
   {
      result.dims[i] = tensor->dims[i+1];
   }

   result.size = tensor->size;
   result.nbytes = tensor->nbytes;
   result.data = tensor->data;
   result.name = tensor->name;

   return result;
}

// NOTE(irwin): contiguous only
static inline TestTensor tensor_index_first_dim( TestTensor *tensor_to_slice, int at_index, b32 keep_dim )
{
   Assert( tensor_is_valid( tensor_to_slice ) );
   Assert( at_index >= 0 );

   int first_dimension_stride = tensor_to_slice->size / tdim(tensor_to_slice, 0);

   TestTensor result = {0};

   int offset = first_dimension_stride * at_index;
   if ( offset < tensor_to_slice->size )
   {
      result.data = tensor_to_slice->data + offset;
      if ( tensor_to_slice->ndim == 1 )
      {
         result.ndim = 1;
         result.dims[0] = tensor_to_slice->dims[0];
      }
      else
      {
         if ( keep_dim )
         {
            result.ndim = tensor_to_slice->ndim;
            for ( int i = 0; i < result.ndim; ++i ) result.dims[i] = tensor_to_slice->dims[i];
            result.dims[0] = 1;
         }
         else
         {
            result.ndim = tensor_to_slice->ndim - 1;
            for ( int i = 0; i < result.ndim; ++i ) result.dims[i] = tensor_to_slice->dims[i + 1];
         }
      }

      result.size = first_dimension_stride;
      result.nbytes = first_dimension_stride * (tensor_to_slice->nbytes / tensor_to_slice->size);
   }

   return result;
}

static inline void zero_tensor( TestTensor *tensor_to_zero )
{
   memset( tensor_to_zero->data, 0, tensor_to_zero->nbytes );
}


static inline TestTensor *tensor_zeros( MemoryArena *arena, int ndim, int dims[] )
{
   TestTensor *result = pushStruct( arena, TestTensor );
   Assert( ndim <= 8 );
   result->ndim = ndim;

   static_assert(sizeof( result->dims[0] ) == sizeof( int ), "ERROR");
   int size = 1;
   for ( int i = 0; i < result->ndim; ++i )
   {
      size *= dims[i];
      result->dims[i] = dims[i];
   }
   result->size = size;
   result->nbytes = size * sizeof( float );
   result->data = pushArray( arena, result->size, float );

   return result;
}

static inline TestTensor *tensor_zeros_1d( MemoryArena *arena, int dim0 )
{
   int dims[1] = {dim0};
   return tensor_zeros( arena, 1, dims );
}

static inline TestTensor *tensor_zeros_2d( MemoryArena *arena, int dim0, int dim1 )
{
   int dims[2] = {dim0, dim1};
   return tensor_zeros( arena, 2, dims );
}

static inline TestTensor *tensor_zeros_3d( MemoryArena *arena, int dim0, int dim1, int dim2 )
{
   int dims[3] = {dim0, dim1, dim2};
   return tensor_zeros( arena, 3, dims );
}

static inline void broadcast_value_to_tensor( TestTensor *tensor, float value )
{
   Assert( tensor_is_valid( tensor ) );

   for ( int data_index = 0; data_index < tensor->size; ++data_index )
   {
      tensor->data[data_index] = value;
   }
}

static inline void tensor_relu_inplace( TestTensor *tensor )
{
   Assert( tensor_is_valid( tensor ) );

   relu_inplace( tensor->data, tensor->size );
}

// TODO(irwin):
// - [x] move to tensor source files
// - [ ] use where applicable
static inline TestTensor *tensor_zeros_like( MemoryArena *arena, TestTensor *reference )
{
   TestTensor *result = pushStruct( arena, TestTensor );
   result->ndim = reference->ndim;

   static_assert(sizeof( result->dims[0] ) == sizeof( int ), "ERROR");
   for ( int i = 0; i < result->ndim; ++i )
   {
      result->dims[i] = reference->dims[i];
   }
   result->nbytes = reference->nbytes;
   result->size = reference->size;
   result->data = pushArray( arena, result->size, float );

   return result;
}

static inline TestTensor *tensor_copy( MemoryArena *arena, TestTensor *reference )
{
   TestTensor *result = tensor_zeros_like( arena, reference );
   memmove(result->data, reference->data, reference->nbytes);

   if (result->name != 0)
   {
      result->name = copyStringToArena( arena, reference->name, 0 );
   }

   return result;
}

static inline void tensor_add_inplace_nd( TestTensor *lhs, TestTensor *rhs )
{
   Assert( lhs->size == rhs->size );
   Assert( lhs->ndim == rhs->ndim );
   Assert( 0 == memcmp( lhs->dims, rhs->dims, lhs->ndim ) );

   add_arrays_inplace( lhs->data, lhs->size, rhs->data );
}


static inline TestTensor *tensor_transpose_last_2d( MemoryArena *arena, TestTensor *source )
{
   Assert( source->ndim >= 2 );

   TestTensor *output = tensor_zeros_like( arena, source );

   output->dims[tdimindex( source, -2 )] = tdim( source, -1 );
   output->dims[tdimindex( source, -1 )] = tdim( source, -2 );


   int columns = tdim( source, -1 );
   int rows = tdim( source, -2 );
   int row_stride = columns;

   int batch_stride = columns * rows;
   int batch_count = source->size / batch_stride;

   for (int batch_index = 0; batch_index < batch_count; ++batch_index)
   {
      float *output_data_batch = output->data + batch_index * batch_stride;
      float *source_data_batch = source->data + batch_index * batch_stride;

      float *data = output_data_batch;

      for ( int x = 0; x < columns; ++x )
      {
         for ( int y = 0; y < rows; ++y )
         {
            float value = source_data_batch[y * row_stride + x];
            *data++ = value;
         }
      }

      Assert( data - output_data_batch == batch_stride );
   }

   return output;
}

static inline void tensor_linear( TestTensor *input,
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

typedef struct ConvOutputShape ConvOutputShape;
struct ConvOutputShape
{
   int batch_size;
   int channels_out;
   int sequence_length;
};

static inline ConvOutputShape conv_output_shape( TestTensor *input, TestTensor *weights, int stride )
{
   Assert( weights->ndim == 3 );

   ConvOutputShape out = {0};
   out.batch_size = input->ndim == 3 ? tdim( input, -3 ) : 1;
   out.channels_out = tdim( weights, 0 );

   int sequence_count_in = tdim(input, -1);
   int kernel_size = tdim(weights, -1);
   int hop_length = stride;
   out.sequence_length = 1 + (sequence_count_in - kernel_size) / hop_length;

   return out;
}

static inline ConvOutputShape conv_output_shape_shape( ConvOutputShape input_shape, TestTensor *weights, int stride )
{
   TestTensor fake_input = {.ndim = 3, .dims = {input_shape.batch_size, input_shape.channels_out, input_shape.sequence_length}};
   
   return conv_output_shape( &fake_input, weights, stride );
}

static inline ConvOutputShape conv_block_output_shape( TestTensor *input, TestTensor *dw_conv_weights, TestTensor *pw_conv_weights )
{
   int sequence_count_in = tdim(input, -1);
   int kernel_size_dw = tdim(dw_conv_weights, -1);
   int dw_pad = 2;

   // dw_out
   int batch_size = input->ndim == 3 ? tdim( input, -3 ) : 1;
   // int channels_out = tdim( dw_conv_weights, 0 );
   int out_sequence_length_dw = 1 + (sequence_count_in + 2 * dw_pad - kernel_size_dw);

   // pw_out
   ConvOutputShape out = {0};
   out.batch_size = batch_size;
   out.channels_out = tdim(pw_conv_weights, 0);
   int kernel_size_pw = tdim(pw_conv_weights, -1);
   out.sequence_length = 1 + (out_sequence_length_dw - kernel_size_pw);

   return out;
}


static inline ConvOutputShape shape_for_transformer( TestTensor *input, TransformerLayer_Weights weights, int stride )
{
   ConvOutputShape conv_block_out_shape = conv_block_output_shape( input, weights.dw_conv_weights, weights.pw_conv_weights );
   return conv_output_shape_shape( conv_block_out_shape, weights.conv_weights, stride );
}

static inline ConvOutputShape shape_for_transformer_shape( ConvOutputShape input_shape, TransformerLayer_Weights weights, int stride )
{
   TestTensor fake_input_tensor = {.ndim = 3, .dims = {input_shape.batch_size, input_shape.channels_out, input_shape.sequence_length}};


   return shape_for_transformer( &fake_input_tensor, weights, stride );
}

static inline ConvOutputShape shape_for_encoder( TestTensor *input, Encoder_Weights encoder_weights )
{
   ConvOutputShape l1_output_required_shape = shape_for_transformer( input, encoder_weights.l1, encoder_weights.l1_conv_stride );
   ConvOutputShape l2_output_required_shape = shape_for_transformer_shape( l1_output_required_shape, encoder_weights.l2, encoder_weights.l2_conv_stride );
   ConvOutputShape l3_output_required_shape = shape_for_transformer_shape( l2_output_required_shape, encoder_weights.l3, encoder_weights.l3_conv_stride );
   ConvOutputShape l4_output_required_shape = shape_for_transformer_shape( l3_output_required_shape, encoder_weights.l4, encoder_weights.l4_conv_stride );
   
   return l4_output_required_shape;
}

static inline TestTensor *tensor_zeros_for_conv( MemoryArena *arena, TestTensor *input, TestTensor *weights, int stride )
{
   ConvOutputShape shape = conv_output_shape( input, weights, stride );
   return tensor_zeros_3d( arena, shape.batch_size, shape.channels_out, shape.sequence_length );
}