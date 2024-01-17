#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define MEMORY_IMPLEMENTATION
#include "memory.h"

#include "decoder.c"

typedef struct TestTensor_Header TestTensor_Header;
struct TestTensor_Header
{
   int version;
   int tensor_count;
};

typedef struct TestTensor
{
   int dummy_;
   // int dummy2_;
   int ndim;
   int *dims;
   int size;
   int nbytes;
   const char *name;
   float *data;
} TestTensor;

// static_assert(sizeof(TestTensor) == 64, "Wrong size");

typedef struct LoadTesttensorResult LoadTesttensorResult;
struct LoadTesttensorResult
{
   int tensor_count;
   TestTensor *tensor_array;
};

LoadTesttensorResult load_testtensor( const char *path )
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();

   LoadTesttensorResult result = { 0 };

   // Assert(tensor);
   // memset(tensor, 0, sizeof(*tensor));

   FILE *f = fopen( path, "rb" );
   AssertMessage( f, "Couldn't open file" );

   TestTensor_Header header = { 0 };
   Assert( fread( &header, sizeof( header ), 1, f ) );
   Assert( header.version == 1 );

   int tensor_count = header.tensor_count;
   Assert( tensor_count > 0 );

   TestTensor *tensor_array = pushArray( debug_arena, tensor_count, TestTensor );

   for ( int i = 0; i < tensor_count; ++i )
   {
      TestTensor *tensor = tensor_array + i;
      int name_len = 0;
      Assert( fread( &name_len, sizeof( name_len ), 1, f ) );
      Assert( name_len );
      char *name = pushSizeZeroed( debug_arena, name_len + 1, 1 );
      Assert( fread( name, sizeof( char ), name_len, f ) );
      tensor->name = name;
   }

   for ( int i = 0; i < tensor_count; ++i )
   {
      TestTensor *tensor = tensor_array + i;

      Assert( fread( &tensor->ndim, sizeof( tensor->ndim ), 1, f ) );
      if ( tensor->ndim )
      {
         tensor->dims = pushArray( debug_arena, tensor->ndim, int );
         Assert( fread( tensor->dims, sizeof( tensor->dims[0] ), tensor->ndim, f ) );
      }
      Assert( fread( &tensor->size, sizeof( tensor->size ), 1, f ) );
      Assert( fread( &tensor->nbytes, sizeof( tensor->nbytes ), 1, f ) );

      tensor->data = pushSizeZeroed( debug_arena, tensor->nbytes, 1 );
      Assert( fread( tensor->data, tensor->nbytes, 1, f ) );
   }

   fclose( f );

   result.tensor_array = tensor_array;
   result.tensor_count = tensor_count;

   Assert( result.tensor_array );
   Assert( result.tensor_count );

   return result;
}


static void print_tensors( LoadTesttensorResult res )
{
   fprintf( stderr, "Tensor count: %d\n", res.tensor_count );
   for ( int i = 0; i < res.tensor_count; ++i )
   {
      TestTensor *t = res.tensor_array + i;
      fprintf( stderr, "%s:\n", t->name );

      fprintf( stderr, "ndim %d\n", t->ndim );
      if ( t->ndim )
      {
         fprintf( stderr, "dims" );
         for ( int ndim = 0; ndim < t->ndim; ++ndim )
         {
            fprintf( stderr, " %d", t->dims[ndim] );
         }
         fprintf( stderr, "\n" );
      }
      fprintf( stderr, "size %d\n", t->size );
      fprintf( stderr, "nbytes %d\n", t->nbytes );
      int cutoff = t->size <= 10 ? t->size : 10;
      for ( int j = 0; j < cutoff; ++j )
      {
         fprintf( stderr, "%f\n", t->data[j] );
      }

      fprintf( stderr, "\n" );
   }
}

static b32 all_close( float *left, float *right, int count, float atol )
{
   for ( int i = 0; i < count; ++i )
   {
      float adiff = fabsf( left[i] - right[i] );
      if ( adiff > atol )
      {
         return 0;
      }
   }

   return 1;
}

// int main(int argc, char *argv[])
int main()
{
   MemoryArena *debug_arena = DEBUG_getDebugArena();

   LoadTesttensorResult res = { 0 };
   res = load_testtensor( "decoder_test.testtensor" );
   TestTensor *input = res.tensor_array + 0;
   TestTensor *weights = res.tensor_array + 1;
   TestTensor *biases = res.tensor_array + 2;
   TestTensor *result = res.tensor_array + 3;

   size_t output_size = input->dims[0] * weights->dims[0] * sizeof( float );
   Assert( output_size == result->nbytes );

   float *output = pushSizeZeroed( debug_arena, output_size, 1 );
   int output_ndims = result->ndim;
   int *output_dims = pushArray( debug_arena, result->ndim, int );

   int result_ok = decoder( input->data, input->dims, input->ndim, weights->data, weights->dims, weights->ndim, biases->data, biases->dims, biases->ndim, output, output_dims, output_ndims );
   Assert( result_ok );

   float atol = 1e-10f;
   b32 pass = all_close( result->data, output, result->size, atol );
   if ( pass )
   {
      fprintf( stderr, "All tests passed!\n" );
   } else
   {
      fprintf( stderr, "Failed test!\n" );
   }

   // Assert(memcmp(output, result->data, output_size) == 0);

   // print_tensors(res);

   return 0;
}
