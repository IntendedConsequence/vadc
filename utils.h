#pragma once
#include <stdint.h>
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


#define Kilobytes(value) ((value) * 1024LL)
#define Megabytes(value) (Kilobytes(value) * 1024LL)
#define Gigabytes(value) (Megabytes(value) * 1024LL)


#ifndef VAR_UNUSED
#define VAR_UNUSED(x) do { (void)sizeof(x); } while(0)
#endif

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

#define Assert(truth) do { if(!(truth)) { ASSERT_ACTION(ASSERT_TOSTRING(truth)); ASSERT_HALT; } } while (0)
#define AssertFail(message) do { ASSERT_ACTION(message); ASSERT_HALT; } while (0)
#define AssertMessage(truth, message) Assert((truth) && message)

#define ArrayCount(array) (sizeof(array) / sizeof((array)[0]))

typedef struct Arena
{
    u8 *base;
    u64 used;
    u64 size;
} Arena;

static u8 debug_arena_buffer[Megabytes(16)];

static struct Arena debug_arena = {.base = &debug_arena_buffer[0], .size=sizeof(debug_arena_buffer)};

static void *arena_push (struct Arena *arena, u64 size)
{
    assert(arena->base);
    assert(size <= arena->size - arena->used);

    u8 *address = arena->base + arena->used;
    arena->used += size;

    return address;
}

static void *arena_pushz (struct Arena *arena, u64 size)
{
    void *address = arena_push(arena, size);
    memset(address, 0, size);

    return address;
}

static void arena_pop (struct Arena *arena, u64 size)
{
    assert(arena->base);
    if (size <= arena->used)
    {
        arena->used -= size;
    }
    else
    {
        arena->used = 0;
    }

}

static void arena_reset (struct Arena *arena)
{
    assert(arena->base);
    arena->used = 0;
}

typedef struct TestTensor_Header
{
    int version;
    int tensor_count;
} TestTensor_Header;

typedef struct TestTensor
{
    int ndim;
    int *dims;
    int size;
    int nbytes;
    const char *name;
    float *data;
} TestTensor;

// static_assert(sizeof(TestTensor) == 64, "Wrong size");

typedef struct LoadTesttensorResult
{
    int tensor_count;
    TestTensor *tensor_array;
} LoadTesttensorResult;

struct LoadTesttensorResult load_testtensor(const char *path)
{
    LoadTesttensorResult result = {0};

    // Assert(tensor);
    // memset(tensor, 0, sizeof(*tensor));

    FILE *f = fopen(path, "rb");
    AssertMessage(f, "Couldn't open file");

    TestTensor_Header header = {0};
    Assert(fread(&header, sizeof(header), 1, f));
    Assert(header.version == 1);

    int tensor_count = header.tensor_count;
    Assert(tensor_count > 0);

    TestTensor *tensor_array = arena_pushz(&debug_arena, sizeof(TestTensor) * tensor_count);

    for (int i = 0; i < tensor_count; ++i)
    {
        TestTensor *tensor = tensor_array + i;
        int name_len = 0;
        Assert(fread(&name_len, sizeof(name_len), 1, f));
        Assert(name_len);
        char *name = arena_pushz(&debug_arena, name_len + 1);
        Assert(fread(name, sizeof(char), name_len, f));
        tensor->name = name;
    }

    for (int i = 0; i < tensor_count; ++i)
    {
        TestTensor *tensor = tensor_array + i;

        Assert(fread(&tensor->ndim, sizeof(tensor->ndim), 1, f));
        if (tensor->ndim)
        {
            tensor->dims = arena_pushz(&debug_arena, tensor->ndim * sizeof(tensor->dims[0]));
            Assert(fread(tensor->dims, sizeof(tensor->dims[0]), tensor->ndim, f));
        }
        Assert(fread(&tensor->size, sizeof(tensor->size), 1, f));
        Assert(fread(&tensor->nbytes, sizeof(tensor->nbytes), 1, f));

        tensor->data = arena_pushz(&debug_arena, tensor->nbytes);
        Assert(fread(tensor->data, tensor->nbytes, 1, f));
    }

    fclose(f);

    result.tensor_array = tensor_array;
    result.tensor_count = tensor_count;

    Assert(result.tensor_array);
    Assert(result.tensor_count);

    return result;
}