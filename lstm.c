#include "utils.h"
#include <math.h>

#if !defined(DEBUG_PRINT)
# define DEBUG_PRINT 0
#endif // DEBUG_PRINT

__declspec(dllexport)
void mytanh (const float *arr, int count, float *out)
{
    for (int i = 0; i < count; ++i)
    {
        out[i] = tanhf(arr[i]);
    }
}

static void mytanh_inplace (float *arr, int count)
{
    for (int i = 0; i < count; ++i)
    {
        float value = arr[i];
        arr[i] = tanhf(value);
    }
}

__declspec(dllexport)
void mysigmoid (const float *arr, int count, float *out)
{
    for (int i = 0; i < count; ++i)
    {
        out[i] = 1.0f / (1.0f + expf(-arr[i]));
    }
}

static void mysigmoid_inplace (float *arr, int count)
{
    for (int i = 0; i < count; ++i)
    {
        float value = arr[i];
        arr[i] = 1.0f / (1.0f + expf(-value));
    }
}

#include "matmul.c"














/*

*/

//#error ##__nito_inline__tmpdir

// void mydot_arrarr (float *arr, int count, float *arr2, int arr2_rows, float *arr_out);

static void add_arrays_inplace (float *array_a, int count, float *array_b)
{
    for (int i = 0; i < count; ++i)
    {
        array_a[i] += array_b[i];
    }
}

static void debugprint_array(const float *arr, int count, FILE *file_out)
{
    for (int i = 0; i < count; ++i)
    {
        fprintf(file_out, "%f\n", arr[i]);
    }
}

#if DEBUG_PRINT
# define DEBUG_ARR_OUT(arr, count, file) do { debugprint_array(arr, count, file) } while(0)
#else
# define DEBUG_ARR_OUT(arr, count, file) do { (void)(sizeof(0)); } while(0)
#endif // DEBUG_PRINT


__declspec(dllexport)
void lstm_cell (const float *input_x, int input_x_count, const float *hidden_state_previous, const float *cell_state_previous, const float *weights_transposed, const float *biases, float *output_hc)
{
#if DEBUG_PRINT
    FILE* debugout = fopen("lstm_debug.txt", "w");
#endif // DEBUG_PRINT

    int combined_count = input_x_count * 2;
    u64 mark = debug_arena.used;
    float *input_and_hidden_state = arena_pushz(&debug_arena, combined_count * sizeof(float));

    // NOTE(irwin): concatenate arrays
    memcpy(input_and_hidden_state, input_x, input_x_count * sizeof(float));
    memcpy(input_and_hidden_state + input_x_count, hidden_state_previous, input_x_count * sizeof(float));
    // DEBUG_ARR_OUT(input_and_hidden_state, combined_count, debugout);

    float *output_array = arena_pushz(&debug_arena, combined_count * 2 * sizeof(float));
    mydot_arrarr(input_and_hidden_state, combined_count, weights_transposed, combined_count * 2, output_array);
    add_arrays_inplace(output_array, combined_count * 2, biases);
    // DEBUG_ARR_OUT(output_array, combined_count * 2, debugout);

    float *input_gates, *forget_gates, *update_gates, *output_gates;
    input_gates = output_array + input_x_count * 0;
    forget_gates = output_array + input_x_count * 1;
    update_gates = output_array + input_x_count * 2;
    output_gates = output_array + input_x_count * 3;

    mysigmoid_inplace(input_gates, input_x_count);
    mysigmoid_inplace(forget_gates, input_x_count);
    mytanh_inplace(update_gates, input_x_count);
    mysigmoid_inplace(output_gates, input_x_count);

    float *h = output_hc + input_x_count * 0;
    float *c = output_hc + input_x_count * 1;

    for (int j = 0; j < input_x_count; ++j)
    {
        c[j] = forget_gates[j] * cell_state_previous[j] + input_gates[j] * update_gates[j];
    }
    mytanh(c, input_x_count, h);

    for (int j = 0; j < input_x_count; ++j)
    {
        h[j] *= output_gates[j];
    }

    debug_arena.used = mark;

#if DEBUG_PRINT
    fclose(debugout);
#endif // DEBUG_PRINT
}

__declspec(dllexport)
void lstm (const float *input_x, int input_x_count, const float *hidden_state_previous, const float *cell_state_previous, const float *weights_transposed, const float *biases, float *output_hc)
{
    // int combined_count = input_x_count * 2;
    int hidden_state_stride = input_x_count;
    int cell_state_stride = input_x_count;
    int weights_stride = (input_x_count * 2) * (input_x_count * 4);
    int biases_stride = (input_x_count * 4);

    u64 mark = debug_arena.used;
    float *output_hc_unordered = arena_pushz(&debug_arena, (hidden_state_stride + cell_state_stride) * 2 * sizeof(float));

    lstm_cell(input_x, input_x_count, hidden_state_previous, cell_state_previous, weights_transposed, biases, output_hc_unordered);
    lstm_cell(output_hc_unordered, input_x_count, hidden_state_previous + hidden_state_stride, cell_state_previous + cell_state_stride, weights_transposed + weights_stride, biases + biases_stride, output_hc_unordered + hidden_state_stride + cell_state_stride);

    // h0,c0 -> h0,h1
    // h1,c1 -> c0,c1

    // h0
    memcpy(output_hc, output_hc_unordered, hidden_state_stride * sizeof(float));
    // h1
    memcpy(output_hc + hidden_state_stride, output_hc_unordered + hidden_state_stride + cell_state_stride, hidden_state_stride * sizeof(float));
    // c0
    memcpy(output_hc + hidden_state_stride + cell_state_stride, output_hc_unordered + hidden_state_stride, cell_state_stride * sizeof(float));
    // c1
    memcpy(output_hc + hidden_state_stride + cell_state_stride + hidden_state_stride, output_hc_unordered + hidden_state_stride + cell_state_stride + hidden_state_stride, cell_state_stride * sizeof(float));

    debug_arena.used = mark;
}
