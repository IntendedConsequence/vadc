// #include <stdlib.h> //malloc, free

#include "utils.h"
#include "tensor.h"
#include "memory.h"
#include "maths.h"


// NOTE(irwin): specialized for kernel_size = 5, padding = 2 (with zeros)
// IMPORTANT(irwin): apparently, expecting weights with flipped kernels
// coincidentally matches PyTorch conv1d implementation, where conv1d is
// implemented actually as cross-correlation. Since convolution is
// cross correlation with weight kernels flipped, PyTorch convolution really
// is cross correlation. Numpy's np.convolve however does flip the kernels,
// so watch out for that.
// It seems that the reason for PyTorch conv1d being this way is just because
// when training a CNN, convolution and cross correlation don't have effect
// on training, but the extra step of flipping the kernels just slows down
// the process unnecessarily.
VADC_API
void convolve_k5_pad2 (const float *arr, int count, const float *kernel_flipped, float *arr_out, float bias)
{
    int kernel_size = 5;
    int padding = 2;
    // NOTE(irwin): for kernel_size = 5, padding = 2, out_array_count equals count
    int out_array_count = count - kernel_size + 1 + padding + padding;

    // NOTE(irwin): since we know that padding is 2 zeros, we can compute first two elements as if we had a kernel
    // of size 4 and 3 for elements at index 0 and 1, respectively, because the padded zeroes effectively mask out
    // the first elements of the kernel.
    arr_out[0] = bias + dotproduct(arr, kernel_size - 2, kernel_flipped + 2, kernel_size - 2);
    arr_out[1] = bias + dotproduct(arr, kernel_size - 1, kernel_flipped + 1, kernel_size - 1);

    for (int i = 0; i < count - kernel_size + 1; ++i)
    {
        float value = dotproduct(arr + i, kernel_size, kernel_flipped, kernel_size);
        arr_out[padding + i] = bias + value;
    }

    // NOTE(irwin): we repeat the same thing for the last two elements as we did for the first two. However,
    // this would mean we need to get the pointer to the last 4 and 3 elements of the input array. This would
    // mean `arr + count - 4` and `arr + count - 3`, or `arr + count - kernel_size + 1`. BUT!
    // If we did that, the calls to dotproduct would look like:
    //
    // ... = dotproduct(arr_pad + 0, kernel_size - 1, kernel_flipped, kernel_size - 1);
    // ... = dotproduct(arr_pad + 1, kernel_size - 2, kernel_flipped, kernel_size - 2);
    // which is harder to read and understand, which offsets do what, from which end etc
    // So to make it more uniform, we instead compute the pointer to the last kernel_size elements of the array,
    // so the offsets are matched now.
    // We do the same thing with arr_out_one_before_two_last_elements following the same principle, with the only
    // difference being we get the pointer to one output array element BEFORE the last two output elements,
    // which we can then offset by the same amount.
    const float *arr_pad = arr + count - kernel_size;
    float *arr_out_one_before_two_last_elements = arr_out + out_array_count - 2 - 1;
    arr_out_one_before_two_last_elements[1] = bias + dotproduct(arr_pad + 1, kernel_size - 1, kernel_flipped, kernel_size - 1);
    arr_out_one_before_two_last_elements[2] = bias + dotproduct(arr_pad + 2, kernel_size - 2, kernel_flipped, kernel_size - 2);
}

static void dw_conv_tensor (TestTensor input, int in_out_channels_groups, TestTensor filters, TestTensor biases, TestTensor output)
{
    Assert(tensor_is_valid(input));
    Assert(tensor_is_valid(filters));
    Assert(tensor_is_valid(biases));
    Assert(tensor_is_valid(output));

    Assert(input.ndim == 2);
    Assert(filters.ndim == 2);
    Assert(biases.ndim == 1);
    Assert(output.ndim == 2);

    Assert(input.dims[0] == in_out_channels_groups);
    Assert(filters.dims[0] == in_out_channels_groups);
    Assert(biases.dims[0] == in_out_channels_groups);
    Assert(output.dims[0] == in_out_channels_groups);

    for (int i = 0; i < in_out_channels_groups; ++i)
    {
        float *arr_out = index2d(output, i, 0);
        float *arr_in = index2d(input, i, 0);
        float *arr_filters = index2d(filters, i, 0);
        float bias = biases.data[i];
        convolve_k5_pad2(arr_in, input.dims[1], arr_filters, arr_out, bias);
    }
}

static void pw_conv_tensor (TestTensor input, TestTensor filters, TestTensor biases, TestTensor output)
{
    Assert(tensor_is_valid(input));
    Assert(tensor_is_valid(filters));
    Assert(tensor_is_valid(biases));
    Assert(tensor_is_valid(output));

    Assert(input.ndim == 2);
    Assert(filters.ndim == 3);
    Assert(biases.ndim == 1);
    Assert(output.ndim == 2);

    int in_channels = input.dims[0];
    int array_count = input.dims[1];
    int out_channels = filters.dims[0];
    int filter_count = out_channels;
    int kernel_size = filters.dims[2];
    int output_array_count = array_count - kernel_size + 1;

    Assert(filters.dims[1] == in_channels);
    Assert(output.dims[0] == filter_count);
    Assert(output.dims[1] == output_array_count);
    Assert(biases.dims[0] == filter_count);

    for (int filter_index = 0; filter_index < filter_count; ++filter_index)
    {

        TestTensor output_filter = tensor_slice_first_dim(output, filter_index);
        broadcast_value_to_tensor(output_filter, biases.data[filter_index]);
        // zero_tensor(output_filter);
        // float *output_arr = index2d(output, filter_index, 0);

        // memset(output_arr, 0, output_array_count * sizeof(float));
        for (int channel_index = 0; channel_index < in_channels; ++channel_index)
        {
            float *kernel = index3d(filters, filter_index, channel_index, 0);

            float *channel = index2d(input, channel_index, 0);
            for (int index = 0; index < array_count; ++index)
            {
                output_filter.data[index] += dotproduct(channel + index, kernel_size, kernel, kernel_size);
            }
        }
    }
}


VADC_API
void convolve_muladd (float *arr, int count, float kernel, float *arr_out)
{
    for (int i = 0; i < count; ++i)
    {
        arr_out[i] += kernel * arr[i];
    }
}

VADC_API
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

VADC_API
void convolve_mc_mf_bias (float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias)
{
    for (int i = 0; i < filter_count; ++i)
    {
        int out_stride = i * array_count;
        convolve_mc(arr, in_channel_count, array_count, kernels + i * in_channel_count, arr_out + out_stride, bias ? bias[i] : 0.0f);
    }
}

VADC_API
void convolve_mc_mf (float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out)
{
    convolve_mc_mf_bias(arr, in_channel_count, array_count, kernels, filter_count, arr_out, 0);
}



VADC_API
void convolve_mc_mf_batch_bias (int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias)
{
    for (int i = 0; i < batch; ++i)
    {
        int stride = i * in_channel_count * array_count;
        int out_stride = i * array_count * filter_count;
        convolve_mc_mf_bias(arr + stride, in_channel_count, array_count, kernels, filter_count, arr_out + out_stride, bias);
    }
}

VADC_API
void convolve_mc_mf_batch (int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out)
{
    convolve_mc_mf_batch_bias(batch, arr, in_channel_count, array_count, kernels, filter_count, arr_out, 0);
}

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
        memcpy(relu_result, input, input_size); // TODO(irwin): memmove?
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

                // TODO(irwin): sigmoid
                mean_result[output_offset++] = 1.0f / (1.0f + expf(-mean_value));
                input_offset += input_dims[2];
            }
        }
    }

    endTemporaryMemory( mark );

    return result_ok;
}
