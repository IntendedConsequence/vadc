#include <stdio.h>
#include <stdlib.h>

#include "utils.h"


// int main(int argc, char *argv[])
int main()
{
    LoadTesttensorResult res = {0};
    res = load_testtensor("test.testtensor");

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

    return 0;
}