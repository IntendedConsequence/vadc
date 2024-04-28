#include "utils.h"
#include "tensor.h"
#include "memory.h"
#include "maths.h"


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
VADC_API
int decoder ( float *input, int *input_dims, int input_ndims, float *weights, int *weights_dims, int weights_ndims, float *biases, int *biases_dims, int biases_ndims, float *output, int *output_dims, int output_ndims )
{
   VAR_UNUSED( biases_dims );
   VAR_UNUSED( output_dims );
   VAR_UNUSED( biases_ndims );
   VAR_UNUSED( output_ndims );
   VAR_UNUSED( weights_ndims );

   int result_ok = 1;

   MemoryArena *debug_arena = DEBUG_getDebugArena();

   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   Assert( input_ndims == 3 );
   Assert( weights_ndims == 3 );
   Assert( biases_ndims == 1 );
   Assert( output_ndims == 3 );

   if ( result_ok )
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

   return result_ok;
}

int decoder_tensor ( TestTensor *input, TestTensor *weights, TestTensor *biases, TestTensor *output )
{
   return decoder( input->data, input->dims, input->ndim,
                   weights->data, weights->dims, weights->ndim,
                   biases->data, biases->dims, biases->ndim,
                   output->data, output->dims, output->ndim );
}
