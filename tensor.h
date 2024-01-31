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
static inline float *index2d(TestTensor tensor, int index0, int index1)
{
   Assert(tensor.ndim == 2);
   int dim0_stride = tensor.dims[tensor.ndim - 1];
   return tensor.data + index0 * dim0_stride + index1;
}

static inline float *index3d(TestTensor tensor, int index0, int index1, int index2)
{
   Assert(tensor.ndim == 3);
   int dim0_stride = tensor.dims[tensor.ndim - 1] * tensor.dims[tensor.ndim - 2];
   int dim1_stride = tensor.dims[tensor.ndim - 1];
   return tensor.data + index0 * dim0_stride + index1 * dim1_stride + index2;
}

static inline b32 tensor_is_valid(TestTensor tensor)
{
   return (!!tensor.data) & (!!tensor.nbytes) & (!!tensor.size) & (!!tensor.ndim) & (!!tensor.dims);
}

// NOTE(irwin): contiguous only
static inline TestTensor tensor_slice_first_dim(TestTensor tensor_to_slice, int at_index)
{
   Assert(tensor_is_valid(tensor_to_slice));
   Assert(at_index >= 0);

   int first_dimension_stride = 1;
   for (int dimension_index = 1; dimension_index < tensor_to_slice.ndim; ++dimension_index)
   {
      first_dimension_stride *= tensor_to_slice.dims[dimension_index];
   }

   TestTensor result = {0};

   int offset = first_dimension_stride * at_index;
   if (offset < tensor_to_slice.size)
   {
      result.data = tensor_to_slice.data + offset;
      if (tensor_to_slice.ndim == 1)
      {
         result.ndim = 1;
         result.dims = tensor_to_slice.dims;
      }
      else
      {
         result.ndim = tensor_to_slice.ndim - 1;
         result.dims = tensor_to_slice.dims + 1;
      }

      result.size = tensor_to_slice.size - offset;
      result.nbytes = tensor_to_slice.nbytes - (tensor_to_slice.nbytes / tensor_to_slice.size) * offset;
   }

   return result;
}

static inline void zero_tensor(TestTensor tensor_to_zero)
{
   memset(tensor_to_zero.data, 0, tensor_to_zero.nbytes);
}


static inline TestTensor *tensor_zeros(MemoryArena *arena, int ndim, int dims[])
{
   TestTensor *result = pushStruct(arena, TestTensor);
   result->ndim = ndim;

   static_assert(sizeof(result->dims[0]) == sizeof(int), "ERROR");
   result->dims = pushArray(arena, result->ndim, int);
   int size = 1;
   for (int i = 0; i < result->ndim; ++i)
   {
      size *= dims[i];
      result->dims[i] = dims[i];
   }
   result->size = size;
   result->nbytes = size * sizeof(float);
   result->data = pushArray(arena, result->size, float);

   return result;
}

static inline TestTensor *tensor_zeros_2d(MemoryArena *arena, int dim0, int dim1)
{
   int dims[2] = { dim0, dim1 };
   return tensor_zeros(arena, 2, dims);
}

static inline void broadcast_value_to_tensor(TestTensor tensor, float value)
{
   Assert(tensor_is_valid(tensor));

   for (int data_index = 0; data_index < tensor.size; ++data_index)
   {
      tensor.data[data_index] = value;
   }
}

static inline void tensor_relu_inplace(TestTensor tensor)
{
   Assert(tensor_is_valid(tensor));

   relu_inplace(tensor.data, tensor.size);
}

// TODO(irwin):
// - [x] move to tensor source files
// - [ ] use where applicable
static inline TestTensor *tensor_zeros_like(MemoryArena *arena, TestTensor *reference)
{
   TestTensor *result = pushStruct(arena, TestTensor);
   result->ndim = reference->ndim;

   static_assert(sizeof(result->dims[0]) == sizeof(int), "ERROR");
   result->dims = pushArray(arena, result->ndim, int);
   for (int i = 0; i < result->ndim; ++i)
   {
      result->dims[i] = reference->dims[i];
   }
   result->nbytes = reference->nbytes;
   result->size = reference->size;
   result->data = pushArray(arena, result->size, float);

   return result;
}