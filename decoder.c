#include <math.h> //expf
// #include <stdlib.h> //malloc, free

#include "utils.h"
#include "memory.h"


static inline float sigmoid_one(float value)
{
    return 1.0f / (1.0f + expf(-value));
}


__declspec(dllexport)
void convolve_muladd (float *arr, int count, float kernel, float *arr_out)
{
    for (int i = 0; i < count; ++i)
    {
        arr_out[i] += kernel * arr[i];
    }
}

__declspec(dllexport)
void convolve_mc (float *arr, int in_channel_count, int array_count, float *kernels, float *arr_out, float bias)
{
    memset(arr_out, 0, array_count * sizeof(float));

    for (int i = 0; i < in_channel_count; ++i)
    {
        int stride = i * array_count;
        convolve_muladd(arr + stride, array_count, kernels[i], arr_out);
    }

    for (int i = 0; i < array_count; ++i)
    {
        arr_out[i] += bias;
    }
}

__declspec(dllexport)
void convolve_mc_mf_bias (float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias)
{
    for (int i = 0; i < filter_count; ++i)
    {
        int out_stride = i * array_count;
        convolve_mc(arr, in_channel_count, array_count, kernels + i * in_channel_count, arr_out + out_stride, bias ? bias[i] : 0.0f);
    }
}

__declspec(dllexport)
void convolve_mc_mf (float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out)
{
    convolve_mc_mf_bias(arr, in_channel_count, array_count, kernels, filter_count, arr_out, 0);
}



__declspec(dllexport)
void convolve_mc_mf_batch_bias (int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias)
{
    for (int i = 0; i < batch; ++i)
    {
        int stride = i * in_channel_count * array_count;
        int out_stride = i * array_count * filter_count;
        convolve_mc_mf_bias(arr + stride, in_channel_count, array_count, kernels, filter_count, arr_out + out_stride, bias);
    }
}

__declspec(dllexport)
void convolve_mc_mf_batch (int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out)
{
    convolve_mc_mf_batch_bias(batch, arr, in_channel_count, array_count, kernels, filter_count, arr_out, 0);
}

__declspec(dllexport)
void relu_inplace (float *arr, int array_count)
{
    for (int i = 0; i < array_count; ++i)
    {
        if (arr[i] < 0.0f)
        {
            arr[i] = 0.0f;
        }
    }
}

__declspec(dllexport)
float mean (float *arr, int arr_count)
{
    float result = 0.0f;
    float divisor = (float)arr_count;
    for (int i = 0; i < arr_count; ++i)
    {
        result += arr[i];
    }

    return result / divisor;
}

// return self.conv1d(x.relu()).mean(axis=2, keepdim=True).sigmoid()
// input [N, 64, 7]
// weight [2, 64, 1]
// bias [2]
__declspec(dllexport)
int decoder (float *input, int *input_dims, int input_ndims, float *weights, int *weights_dims, int weights_ndims, float *biases, int *biases_dims, int biases_ndims, float *output, int *output_dims, int output_ndims)
{
    VAR_UNUSED(biases_dims);
    VAR_UNUSED(output_dims);

    int result_ok = 1;

    MemoryArena *debug_arena = DEBUG_getDebugArena();

    TemporaryMemory mark = beginTemporaryMemory( debug_arena );

    assert(input_ndims == 3);
    assert(weights_ndims == 3);
    assert(biases_ndims == 1);
    assert(output_ndims == 3);

    if (result_ok)
    {
        int input_count = 1;
        for (int i = 0; i < input_ndims; ++i)
        {
            input_count *= input_dims[i];
        }

        int input_size = input_count * sizeof(float);

        float *relu_result = pushArray( debug_arena, input_count, float );
        memcpy(relu_result, input, input_size);
        relu_inplace(relu_result, input_count);

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


        float *convolve_result = pushArray(debug_arena, convolve_result_count, float);
        convolve_mc_mf_batch_bias(batch_count, relu_result, input_dims[1], input_dims[2], weights, weights_dims[0], convolve_result, biases);

        // return __LINE__;
        // [N, 2, 1]
        int mean_output_count = 1;
        mean_output_count *= batch_count;
        mean_output_count *= weights_dims[0];

        // float *mean_result = vadc_malloc(mean_output_count * sizeof(float));
        float *mean_result = output;

        int input_offset = 0;
        int output_offset = 0;
        for (int b = 0; b < batch_count; ++b)
        {
            for (int f = 0; f < weights_dims[0]; ++f)
            {
                float mean_value = mean(convolve_result + input_offset, input_dims[2]);
                mean_result[output_offset++] = 1.0f / (1.0f + expf(-mean_value));
                input_offset += input_dims[2];
            }
        }
    }

    endTemporaryMemory( mark );

    return result_ok;
}