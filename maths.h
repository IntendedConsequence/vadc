#ifndef MATHS_INCLUDE_H
#define MATHS_INCLUDE_H

#include "utils.h"
#include <math.h> //expf

static inline float sigmoid_one( float value )
{
   return 1.0f / (1.0f + expf( -value ));
}


VADC_API
void relu_inplace ( float *arr, int array_count );

VADC_API
float mean ( float *arr, int arr_count );

VADC_API
void mytanh ( const float *arr, int count, float *out );


static void mytanh_inplace ( float *arr, int count );


VADC_API
void mysigmoid ( const float *arr, int count, float *out );

static void mysigmoid_inplace ( float *arr, int count );

static void add_arrays ( const float *array_a, int count, const float *array_b, float *array_out );
static void add_arrays_inplace ( float *array_a, int count, const float *array_b );


// row1:     row2:
// [a b c]   [d e f]
//
// result:
// ad + be + cf

VADC_API
float dotproduct ( const float *arr, int count, const float *arr2, int count2 );

// mat1_row:    mat2_transposed:
// [a b c]      [j l n]
//              [k m o]
//
// result:
// [aj+bl+cn ak+bm+co]

VADC_API
void mydot_arrarr ( const float *arr, int count, const float *arr2, int arr2_rows, float *arr_out );

// mata:        matb:
// [a b c]      [j k]
// [d e f]  x   [l m]
// [g h i]      [n o]
//
// result:
// [aj+bl+cn ak+bm+co]
// [dj+el+fn dk+em+fo]
// [gj+hl+in gk+hm+io]
VADC_API
void mymatmul ( float *mata, int mata_rows, int mata_cols, float *matb_transposed, int matb_transposed_rows, int matb_transposed_cols, float *out_result );


VADC_API
void convolve_muladd ( float *arr, int count, float kernel, float *arr_out );

VADC_API
void convolve_mc ( float *arr, int in_channel_count, int array_count, float *kernels, float *arr_out, float bias );

VADC_API
void convolve_mc_mf_bias ( float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias );

VADC_API
void convolve_mc_mf ( float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out );


VADC_API
void convolve_mc_mf_batch_bias ( int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias );

VADC_API
void convolve_mc_mf_batch ( int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out );


#endif // MATHS_INCLUDE_H


#if defined(MATHS_IMPLEMENTATION)

VADC_API
void relu_inplace ( float *arr, int array_count )
{
   for ( int i = 0; i < array_count; ++i )
   {
      if ( arr[i] < 0.0f )
      {
         arr[i] = 0.0f;
      }
   }
}

VADC_API
float mean ( float *arr, int arr_count )
{
   float result = 0.0f;
   float divisor = (float)arr_count;
   for ( int i = 0; i < arr_count; ++i )
   {
      result += arr[i];
   }

   return result / divisor;
}

// row1:     row2:
// [a b c]   [d e f]
//
// result:
// ad + be + cf

VADC_API
float dotproduct ( const float *arr, int count, const float *arr2, int count2 )
{
   int mincount = count > count2 ? count2 : count;

   float result = 0.0f;
   for ( int i = 0; i < mincount; ++i )
   {
      float value = arr[i] * arr2[i];
      result += value;
   }

   return result;
}

// mat1_row:    mat2_transposed:
// [a b c]      [j l n]
//              [k m o]
//
// result:
// [aj+bl+cn ak+bm+co]

VADC_API
void mydot_arrarr ( const float *arr, int count, const float *arr2, int arr2_rows, float *arr_out )
{
   for ( int i = 0; i < arr2_rows; ++i )
   {
      float value = dotproduct( arr, count, arr2 + i * count, count );
      arr_out[i] = value;
   }
}

// mata:        matb:
// [a b c]      [j k]
// [d e f]  x   [l m]
// [g h i]      [n o]
//
// result:
// [aj+bl+cn ak+bm+co]
// [dj+el+fn dk+em+fo]
// [gj+hl+in gk+hm+io]
VADC_API
void mymatmul ( float *mata, int mata_rows, int mata_cols, float *matb_transposed, int matb_transposed_rows, int matb_transposed_cols, float *out_result )
{
   VAR_UNUSED( matb_transposed_cols );
   int mata_stride = mata_cols;
   int out_stride = matb_transposed_rows;

   for ( int i = 0; i < mata_rows; ++i )
   {
      mydot_arrarr( mata + i * mata_stride, mata_cols, matb_transposed, matb_transposed_rows, out_result + i * out_stride );
   }
}

VADC_API
void mytanh ( const float *arr, int count, float *out )
{
   for ( int i = 0; i < count; ++i )
   {
      out[i] = tanhf( arr[i] );
   }
}

static void mytanh_inplace ( float *arr, int count )
{
   for ( int i = 0; i < count; ++i )
   {
      float value = arr[i];
      arr[i] = tanhf( value );
   }
}

VADC_API
void mysigmoid ( const float *arr, int count, float *out )
{
   for ( int i = 0; i < count; ++i )
   {
      out[i] = 1.0f / (1.0f + expf( -arr[i] ));
   }
}

static void mysigmoid_inplace ( float *arr, int count )
{
   for ( int i = 0; i < count; ++i )
   {
      float value = arr[i];
      arr[i] = 1.0f / (1.0f + expf( -value ));
   }
}

static void add_arrays ( const float *array_a, int count, const float *array_b, float *array_out )
{
   for ( int i = 0; i < count; ++i )
   {
      array_out[i] = array_a[i] + array_b[i];
   }
}

static void add_arrays_inplace ( float *array_a, int count, const float *array_b )
{
   for ( int i = 0; i < count; ++i )
   {
      array_a[i] += array_b[i];
   }
}

VADC_API
void convolve_muladd ( float *arr, int count, float kernel, float *arr_out )
{
   for ( int i = 0; i < count; ++i )
   {
      arr_out[i] += kernel * arr[i];
   }
}

VADC_API
void convolve_mc ( float *arr, int in_channel_count, int array_count, float *kernels, float *arr_out, float bias )
{
   memset( arr_out, 0, array_count * sizeof( float ) );

   for ( int i = 0; i < in_channel_count; ++i )
   {
      int stride = i * array_count;
      convolve_muladd( arr + stride, array_count, kernels[i], arr_out );
   }

   for ( int i = 0; i < array_count; ++i )
   {
      arr_out[i] += bias;
   }
}

VADC_API
void convolve_mc_mf_bias ( float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias )
{
   for ( int i = 0; i < filter_count; ++i )
   {
      int out_stride = i * array_count;
      convolve_mc( arr, in_channel_count, array_count, kernels + i * in_channel_count, arr_out + out_stride, bias ? bias[i] : 0.0f );
   }
}

VADC_API
void convolve_mc_mf ( float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out )
{
   convolve_mc_mf_bias( arr, in_channel_count, array_count, kernels, filter_count, arr_out, 0 );
}



VADC_API
void convolve_mc_mf_batch_bias ( int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out, float *bias )
{
   for ( int i = 0; i < batch; ++i )
   {
      int stride = i * in_channel_count * array_count;
      int out_stride = i * array_count * filter_count;
      convolve_mc_mf_bias( arr + stride, in_channel_count, array_count, kernels, filter_count, arr_out + out_stride, bias );
   }
}

VADC_API
void convolve_mc_mf_batch ( int batch, float *arr, int in_channel_count, int array_count, float *kernels, int filter_count, float *arr_out )
{
   convolve_mc_mf_batch_bias( batch, arr, in_channel_count, array_count, kernels, filter_count, arr_out, 0 );
}

#endif //MATHS_IMPLEMENTATION