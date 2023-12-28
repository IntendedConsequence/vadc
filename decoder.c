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
void convolve_mc (float *arr, int in_channel_count, int array_count, float *kernels, float *arr_out)
{
    memset(arr_out, 0, array_count * sizeof(float));

    for (int i = 0; i < in_channel_count; ++i)
    {
        int stride = i * array_count;
        convolve_muladd(arr + stride, array_count, kernels[i], arr_out);
    }
}

__declspec(dllexport)
void convolve_mc_mf (float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out)
{
    for (int i = 0; i < filter_count; ++i)
    {
        int out_stride = i * array_count;
        convolve_mc(arr, in_channel_count, array_count, kernels + i * in_channel_count, arr_out + out_stride);
    }
}


__declspec(dllexport)
void convolve_mc_mf_batch (int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out)
{
    for (int i = 0; i < batch; ++i)
    {
        int stride = i * in_channel_count * array_count;
        int out_stride = i * array_count * filter_count;
        convolve_mc_mf(arr + stride, in_channel_count, array_count, kernels, filter_count, arr_out + out_stride);
    }
}
