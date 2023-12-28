#include <string.h>

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
