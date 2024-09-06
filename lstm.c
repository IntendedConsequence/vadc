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
static inline void lstm_cell ( MemoryArena *arena,
                              const float *input_x,
                              int input_x_count,
                              const float *hidden_state_previous,
                              const float *cell_state_previous,
                              const float *weights_transposed,
                              const float *biases,
                              float *output_h,
                              float *output_c )
{
   TracyCZone(lstm_cell, true);

#if DEBUG_PRINT
   FILE *debugout = fopen( "lstm_debug.txt", "w" );
#endif // DEBUG_PRINT

   MemoryArena *debug_arena = arena;
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

   float *h = output_h;
   float *c = output_c;

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
static inline void lstm ( MemoryArena *arena,
                          const float *input_x,
                          int input_x_count,
                          const float *hidden_state_previous,
                          const float *cell_state_previous,
                          const float *weights_transposed,
                          const float *biases,
                          float *output_h,
                          float *output_c,
                          int layers )
{
   TracyCZone(lstm, true);


   // int combined_count = input_x_count * 2;
   int hidden_state_stride = input_x_count;
   int cell_state_stride = input_x_count;
   int weights_stride = (input_x_count * 2) * (input_x_count * 4);
   int biases_stride = (input_x_count * 4);

   TemporaryMemory mark = beginTemporaryMemory( arena );

   {
      const float *input = input_x;

      // float *output_h = output_hc;
      // float *output_c = output_hc + layers * hidden_state_stride;

      for (int layer_index = 0; layer_index < layers; ++layer_index)
      {
         lstm_cell( arena,
                    input,
                    input_x_count,
                    hidden_state_previous + layer_index * hidden_state_stride,
                    cell_state_previous + layer_index * cell_state_stride,
                    weights_transposed + layer_index * weights_stride,
                    biases + layer_index * biases_stride,
                    output_h,
                    output_c );

         input = output_h;

         output_h += hidden_state_stride;
         output_c += cell_state_stride;
      }
   }

   endTemporaryMemory( mark );
   TracyCZoneEnd(lstm);
}

// IMPORTANT(irwin): biases are expected to be shared for both input data and hidden state. Since pytorch uses separate biases
// for input data and hidden state for CUDA compatibility, if the caller comes from PyTorch, the caller must take care of
// adding the pytorch separate biases (within each lstm cell) before calling this function.
// output:
// [seq, input_x_count], h0,h1, c0,c1
static inline void lstm_seq ( MemoryArena *arena,
                              const float *input_x,
                              int input_x_seq_count,
                              int input_x_count,
                              const float *hidden_state_previous,
                              const float *cell_state_previous,
                              const float *weights_transposed,
                              const float *biases,
                              float *output,
                              int layers )
{
   TracyCZone(lstm_seq, true);

   int input_size = input_x_count;
   int hidden_size = input_x_count;

   TemporaryMemory mark = beginTemporaryMemory( arena );

   // NOTE(irwin): double buffered
   float *input_hc = pushArray( arena, (input_size + hidden_size) * layers, float );
   float *output_hc = pushArray( arena, (input_size + hidden_size) * layers, float );

   const float *input_h = hidden_state_previous;
   const float *input_c = cell_state_previous;

   float *output_h = output_hc;
   float *output_c = output_hc + layers * hidden_size;

   for ( int i = 0; i < input_x_seq_count; ++i )
   {
      lstm( arena,
            input_x + i * input_x_count,
            input_x_count,
            input_h,
            input_c,
            weights_transposed,
            biases,
            output_h,
            output_c,
            layers );

      memmove( output + i * input_x_count, output_h + hidden_size * (layers - 1), hidden_size * sizeof( float ) );

      // NOTE(irwin): swap buffers
      {
         float *temp_swap = input_hc;
         input_hc = output_hc;
         output_hc = temp_swap;

         input_h = input_hc;
         input_c = input_hc + layers * hidden_size;

         output_h = output_hc;
         output_c = output_hc + layers * hidden_size;
      }
   }

   // NOTE(irwin): we read from input_hc because it was just written to by the last iteration and was just flipped
   memmove( output + input_x_seq_count * input_x_count, input_hc, (input_size + hidden_size) * layers * sizeof( float ) );

   endTemporaryMemory( mark );
   TracyCZoneEnd(lstm_seq);
}
