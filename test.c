#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#include "decoder.c"


static void print_tensors(LoadTesttensorResult res)
{
    fprintf(stderr, "Tensor count: %d\n", res.tensor_count);
    for (int i = 0; i < res.tensor_count; ++i)
    {
        TestTensor *t = res.tensor_array + i;
        fprintf(stderr, "%s:\n", t->name);

        fprintf(stderr, "ndim %d\n", t->ndim);
        if (t->ndim)
        {
            fprintf(stderr, "dims");
            for (int ndim = 0; ndim < t->ndim; ++ndim)
            {
                fprintf(stderr, " %d", t->dims[ndim]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "size %d\n", t->size);
        fprintf(stderr, "nbytes %d\n", t->nbytes);
        int cutoff = t->size <= 10 ? t->size : 10;
        for (int j = 0; j < cutoff; ++j)
        {
            fprintf(stderr, "%f\n", t->data[j]);
        }

        fprintf(stderr, "\n");
    }
}

static b32 all_close(float *left, float *right, int count, float atol)
{
    for (int i = 0; i < count; ++i)
    {
        float adiff = fabsf(left[i] - right[i]);
        if (adiff > atol)
        {
            return 0;
        }
    }

    return 1;
}

// int main(int argc, char *argv[])
int main()
{
    LoadTesttensorResult res = {0};
    res = load_testtensor("decoder_test.testtensor");
    TestTensor *input = res.tensor_array + 0;
    TestTensor *weights = res.tensor_array + 1;
    TestTensor *biases = res.tensor_array + 2;
    TestTensor *result = res.tensor_array + 3;

    size_t output_size = input->dims[0] * weights->dims[0] * sizeof(float);
    Assert(output_size == result->nbytes);

    float *output = arena_pushz(&debug_arena, output_size);
    int output_ndims = result->ndim;
    int *output_dims = arena_pushz(&debug_arena, result->ndim * sizeof(int));

    int result_ok = decoder(input->data, input->dims, input->ndim, weights->data, weights->dims, weights->ndim, biases->data, biases->dims, biases->ndim, output, output_dims, output_ndims);
    Assert(result_ok);

    float atol = 1e-10f;
    b32 pass = all_close(result->data, output, result->size, atol);
    if (pass)
    {
        fprintf(stderr, "All tests passed!\n");
    }
    else
    {
        fprintf(stderr, "Failed test!\n");
    }

    // Assert(memcmp(output, result->data, output_size) == 0);

    // print_tensors(res);

    return 0;
}
