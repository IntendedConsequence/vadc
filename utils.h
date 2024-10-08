#pragma once
#include <stdint.h>
#include <stdbool.h> //true, false
#include <stdio.h> //fprintf, stderr
#include <string.h> //memcpy, memset
#include <assert.h> //assert

typedef uint8_t u8;
typedef int8_t s8;

typedef uint16_t u16;
typedef int16_t s16;

typedef uint32_t u32;
typedef int32_t s32;

typedef uint64_t u64;
typedef int64_t s64;

typedef float f32;
typedef double f64;

typedef int32_t b32;

#define mymin(val_a, val_b) ((val_a) < (val_b) ? (val_a) : (val_b))
#define mymax(val_a, val_b) ((val_a) > (val_b) ? (val_a) : (val_b))

#define Kilobytes(value) ((value) * 1024LL)
#define Megabytes(value) (Kilobytes(value) * 1024LL)
#define Gigabytes(value) (Megabytes(value) * 1024LL)


#ifndef VAR_UNUSED
#define VAR_UNUSED(x) do { (void)sizeof(x); } while(0)
#endif

#define VADC_STRINGIFY_(x) #x
#define VADC_TOSTRING(x) VADC_STRINGIFY_(x)

#define VADC_CAT__(a, b) a ## b
#define VADC_CAT_(a, b) VADC_CAT__(a, b)
#define VADC_CAT(a, b) VADC_CAT_(a, b)

// TODO(irwin): define that disables asserts

#ifndef ASSERT_HALT
#define ASSERT_HALT do { __debugbreak(); } while(0)
#endif

#ifndef ASSERT_UNUSED
#define ASSERT_UNUSED(x) VAR_UNUSED(x)
#endif

#ifndef ASSERT_ACTION
// #define ASSERT_ACTION(x) ASSERT_UNUSED(x)
#define ASSERT_ACTION(x) do { fprintf(stderr, "Assertion error in file %s:%d: %s\n", __FILE__, __LINE__, x); } while(0)
#endif

#define ASSERT_TOSTRING_M(x) #x
#define ASSERT_TOSTRING(x) ASSERT_TOSTRING_M(x)

#if !defined(NDEBUG)
# define Assert(truth) do { if(!(truth)) { ASSERT_ACTION(ASSERT_TOSTRING(truth)); ASSERT_HALT; } } while (0)
# define AssertFail(message) do { ASSERT_ACTION(message); ASSERT_HALT; } while (0)
# define AssertMessage(truth, message) Assert((truth) && message)
#else // !defined(NDEBUG)
# define Assert(truth) VAR_UNUSED(truth)
# define AssertFail(message) VAR_UNUSED(message)
# define AssertMessage(truth, message) Assert((truth) && message)
#endif // !defined(NDEBUG)

#define ArrayCount(arr) (sizeof(arr) / sizeof((arr)[0]))
