#pragma once

typedef struct TestTensor TestTensor;

struct TestTensor
{
   // int dummy_;
   // int dummy2_;
   int ndim;
   int *dims;
   int size;
   int nbytes;
   const char *name;
   float *data;
};

// static_assert(sizeof(TestTensor) == 64, "Wrong size");
