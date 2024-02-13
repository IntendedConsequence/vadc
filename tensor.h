#pragma once
#include "maths.h"
#include "memory.h"

typedef struct TestTensor TestTensor;

struct TestTensor
{
   // int dummy_;
   // int dummy2_;
   int ndim;
   int *dims;
   int size;
   int nbytes;
   const char *name;
   float *data;
};

// static_assert(sizeof(TestTensor) == 64, "Wrong size");

// TODO(irwin):
// - [x] move to tensor source files
// - [ ] use where applicable
static inline float *index2d( TestTensor *tensor, int index0, int index1 )
{
   Assert( tensor->ndim == 2 );
   int dim0_stride = tensor->dims[tensor->ndim - 1];
   return tensor->data + index0 * dim0_stride + index1;
}

static inline float *index3d( TestTensor *tensor, int index0, int index1, int index2 )
{
   Assert( tensor->ndim == 3 );
   int dim0_stride = tensor->dims[tensor->ndim - 1] * tensor->dims[tensor->ndim - 2];
   int dim1_stride = tensor->dims[tensor->ndim - 1];
   return tensor->data + index0 * dim0_stride + index1 * dim1_stride + index2;
}

static inline b32 tensor_is_valid( TestTensor *tensor )
{
   return (!!tensor->data) & (!!tensor->nbytes) & (!!tensor->size) & (!!tensor->ndim) & (!!tensor->dims);
}

// NOTE(irwin): contiguous only
static inline TestTensor tensor_slice_first_dim( TestTensor *tensor_to_slice, int at_index )
{
   Assert( tensor_is_valid( tensor_to_slice ) );
   Assert( at_index >= 0 );

   int first_dimension_stride = 1;
   for ( int dimension_index = 1; dimension_index < tensor_to_slice->ndim; ++dimension_index )
   {
      first_dimension_stride *= tensor_to_slice->dims[dimension_index];
   }

   TestTensor result = {0};

   int offset = first_dimension_stride * at_index;
   if ( offset < tensor_to_slice->size )
   {
      result.data = tensor_to_slice->data + offset;
      if ( tensor_to_slice->ndim == 1 )
      {
         result.ndim = 1;
         result.dims = tensor_to_slice->dims;
      }
      else
      {
         result.ndim = tensor_to_slice->ndim - 1;
         result.dims = tensor_to_slice->dims + 1;
      }

      result.size = tensor_to_slice->size - offset;
      result.nbytes = tensor_to_slice->nbytes - (tensor_to_slice->nbytes / tensor_to_slice->size) * offset;
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
   result->ndim = ndim;

   static_assert(sizeof( result->dims[0] ) == sizeof( int ), "ERROR");
   result->dims = pushArray( arena, result->ndim, int );
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

static inline TestTensor *tensor_zeros_2d( MemoryArena *arena, int dim0, int dim1 )
{
   int dims[2] = {dim0, dim1};
   return tensor_zeros( arena, 2, dims );
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
   result->dims = pushArray( arena, result->ndim, int );
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

static inline void tensor_add_inplace( TestTensor *lhs, TestTensor *rhs )
{
   Assert( lhs->size == rhs->size );
   Assert( lhs->ndim == rhs->ndim );
   Assert( 0 == memcmp( lhs->dims, rhs->dims, lhs->ndim ) );

   add_arrays_inplace( lhs->data, lhs->size, rhs->data );
}

static inline TestTensor *tensor_transpose_2d( MemoryArena *arena, TestTensor *source )
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