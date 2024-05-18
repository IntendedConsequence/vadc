#include "utils.h"

// #define MEMORY_IMPLEMENTATION
#include "memory.h"

#include "maths.h"

#if !defined(DEBUG_PRINT)
# define DEBUG_PRINT 0
#endif // DEBUG_PRINT



static void debugprint_array( const float *arr, int count, FILE *file_out )
{
   for ( int i = 0; i < count; ++i )
   {
      fprintf( file_out, "%f\n", arr[i] );
   }
}

#if DEBUG_PRINT
# define DEBUG_ARR_OUT(arr, count, file) do { debugprint_array(arr, count, file) } while(0)
#else
# define DEBUG_ARR_OUT(arr, count, file) do { (void)(sizeof(0)); } while(0)
#endif // DEBUG_PRINT

// IMPORTANT(irwin): biases are expected to be shared for both input data and hidden state. Since pytorch uses separate biases
// for input data and hidden state for CUDA compatibility, if the caller comes from PyTorch, the caller must take care of
// adding the pytorch separate biases before calling this function.
VADC_API
void lstm_cell ( const float *input_x, int input_x_count, const float *hidden_state_previous, const float *cell_state_previous, const float *weights_transposed, const float *biases, float *output_hc )
{
   TracyCZone(lstm_cell, true);

#if DEBUG_PRINT
   FILE *debugout = fopen( "lstm_debug.txt", "w" );
#endif // DEBUG_PRINT

   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   int combined_count = input_x_count * 2;

   float *input_and_hidden_state = pushArray( debug_arena, combined_count, float );

   // NOTE(irwin): concatenate arrays
   memcpy( input_and_hidden_state, input_x, input_x_count * sizeof( float ) );
   memcpy( input_and_hidden_state + input_x_count, hidden_state_previous, input_x_count * sizeof( float ) );
   // DEBUG_ARR_OUT(input_and_hidden_state, combined_count, debugout);

   float *output_array = pushArray( debug_arena, combined_count * 2, float );
   mydot_arrarr( input_and_hidden_state, combined_count, weights_transposed, combined_count * 2, output_array );
   add_arrays_inplace( output_array, combined_count * 2, biases );
   // DEBUG_ARR_OUT(output_array, combined_count * 2, debugout);

   float *input_gates, *forget_gates, *update_gates, *output_gates;
   input_gates = output_array + input_x_count * 0;
   forget_gates = output_array + input_x_count * 1;
   update_gates = output_array + input_x_count * 2;
   output_gates = output_array + input_x_count * 3;

   mysigmoid_inplace( input_gates, input_x_count );
   mysigmoid_inplace( forget_gates, input_x_count );
   mytanh_inplace( update_gates, input_x_count );
   mysigmoid_inplace( output_gates, input_x_count );

   float *h = output_hc + input_x_count * 0;
   float *c = output_hc + input_x_count * 1;

   for ( int j = 0; j < input_x_count; ++j )
   {
      c[j] = forget_gates[j] * cell_state_previous[j] + input_gates[j] * update_gates[j];
   }
   mytanh( c, input_x_count, h );

   for ( int j = 0; j < input_x_count; ++j )
   {
      h[j] *= output_gates[j];
   }

   endTemporaryMemory( mark );

#if DEBUG_PRINT
   fclose( debugout );
#endif // DEBUG_PRINT
   TracyCZoneEnd(lstm_cell);
}

// IMPORTANT(irwin): biases are expected to be shared for both input data and hidden state. Since pytorch uses separate biases
// for input data and hidden state for CUDA compatibility, if the caller comes from PyTorch, the caller must take care of
// adding the pytorch separate biases (within each lstm cell) before calling this function.
VADC_API
void lstm ( const float *input_x, int input_x_count, const float *hidden_state_previous, const float *cell_state_previous, const float *weights_transposed, const float *biases, float *output_hc )
{
   TracyCZone(lstm, true);


   // int combined_count = input_x_count * 2;
   int hidden_state_stride = input_x_count;
   int cell_state_stride = input_x_count;
   int weights_stride = (input_x_count * 2) * (input_x_count * 4);
   int biases_stride = (input_x_count * 4);

   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   float *output_hc_unordered = pushArray( debug_arena, (hidden_state_stride + cell_state_stride) * 2, float );

   lstm_cell( input_x, input_x_count, hidden_state_previous, cell_state_previous, weights_transposed, biases, output_hc_unordered );
   lstm_cell( output_hc_unordered, input_x_count, hidden_state_previous + hidden_state_stride, cell_state_previous + cell_state_stride, weights_transposed + weights_stride, biases + biases_stride, output_hc_unordered + hidden_state_stride + cell_state_stride );

   // h0,c0 -> h0,h1
   // h1,c1 -> c0,c1

   // TODO(irwin): memmove?
   // h0
   memcpy( output_hc, output_hc_unordered, hidden_state_stride * sizeof( float ) );
   // h1
   memcpy( output_hc + hidden_state_stride, output_hc_unordered + hidden_state_stride + cell_state_stride, hidden_state_stride * sizeof( float ) );
   // c0
   memcpy( output_hc + hidden_state_stride + cell_state_stride, output_hc_unordered + hidden_state_stride, cell_state_stride * sizeof( float ) );
   // c1
   memcpy( output_hc + hidden_state_stride + cell_state_stride + hidden_state_stride, output_hc_unordered + hidden_state_stride + cell_state_stride + hidden_state_stride, cell_state_stride * sizeof( float ) );

   endTemporaryMemory( mark );
   TracyCZoneEnd(lstm);
}

// IMPORTANT(irwin): biases are expected to be shared for both input data and hidden state. Since pytorch uses separate biases
// for input data and hidden state for CUDA compatibility, if the caller comes from PyTorch, the caller must take care of
// adding the pytorch separate biases (within each lstm cell) before calling this function.
// output:
// [seq, input_x_count], h0,h1, c0,c1
VADC_API
void lstm_seq ( const float *input_x, int input_x_seq_count, int input_x_count, const float *hidden_state_previous, const float *cell_state_previous, const float *weights_transposed, const float *biases, float *output )
{
   TracyCZone(lstm_seq, true);

   int input_size = input_x_count;
   int hidden_size = input_x_count;

   MemoryArena *debug_arena = DEBUG_getDebugArena();
   TemporaryMemory mark = beginTemporaryMemory( debug_arena );

   float *input_hc = pushArray( debug_arena, (input_size + hidden_size) * 2, float );
   float *input_h = input_hc;
   float *input_c = input_hc + (hidden_size) * 2;
   // TODO(irwin): memmove?
   memcpy( input_h, hidden_state_previous, (hidden_size) * 2 * sizeof( float ) );
   memcpy( input_c, cell_state_previous, (hidden_size) * 2 * sizeof( float ) );

   float *output_hc = pushArray( debug_arena, (input_size + hidden_size) * 2, float );
   for ( int i = 0; i < input_x_seq_count; ++i )
   {
      lstm( input_x + i * input_x_count, input_x_count, input_h, input_c, weights_transposed, biases, output_hc );

      // TODO(irwin): memmove?
      memcpy( input_hc, output_hc, (input_size + hidden_size) * 2 * sizeof( float ) );
      memcpy( output + i * input_x_count, output_hc + hidden_size, hidden_size * sizeof( float ) );
   }
   // TODO(irwin): memmove?
   memcpy( output + input_x_seq_count * input_x_count, output_hc, (input_size + hidden_size) * 2 * sizeof( float ) );

   endTemporaryMemory( mark );
   TracyCZoneEnd(lstm_seq);
}
