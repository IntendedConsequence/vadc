// row1:     row2:
// [a b c]   [d e f]
//
// result:
// ad + be + cf

__declspec(dllexport)
float dotproduct (const float *arr, int count, const float *arr2, int count2)
{
    int mincount = count > count2 ? count2 : count;

    float result = 0.0f;
    for (int i = 0; i < mincount; ++i)
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

__declspec(dllexport)
void mydot_arrarr (const float *arr, int count, const float *arr2, int arr2_rows, float *arr_out)
{
    for (int i = 0; i < arr2_rows; ++i)
    {
        float value = dotproduct(arr, count, arr2 + i * count, count);
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
__declspec(dllexport)
void mymatmul (float *mata, int mata_rows, int mata_cols, float *matb_transposed, int matb_transposed_rows, int matb_transposed_cols, float *out_result)
{
    VAR_UNUSED(matb_transposed_cols);
    int mata_stride = mata_cols;
    int out_stride = matb_transposed_rows;

    for (int i = 0; i < mata_rows; ++i)
    {
        mydot_arrarr(mata + i * mata_stride, mata_cols, matb_transposed, matb_transposed_rows, out_result + i * out_stride);
    }
}
