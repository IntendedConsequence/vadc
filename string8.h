#pragma once
#include "utils.h"

#include "memory.h"

#include <stdarg.h> //va_list, va_start, va_end

typedef s64 strSize;

typedef struct String8 String8;
struct String8
{
    const s8 *begin;
    strSize size;
};

String8 Widechar_ToString8(MemoryArena *arena, const wchar_t *str, size_t str_length);
int String8_ToWidechar(MemoryArena *arena, wchar_t **dest, String8 source);

String8 String8FromPointerSize(const s8 *pointer, strSize size);
String8 String8FromRange(const s8 *first, const s8 *one_past_last);
String8 String8FromCString(const char *cstring);

// NOTE(irwin): copies the string to arena ensuring a null terminator
String8 String8ToCString(MemoryArena *arena, String8 source_string);
inline String8 String8FromWidechar(MemoryArena *arena, const wchar_t *str, size_t str_length)
{
    return Widechar_ToString8(arena, str, str_length);
}

#define String8FromLiteral(value) { .begin = (value), .size = sizeof(value) - 1 }

String8 String8_pushfv(MemoryArena *arena, const char *format, va_list args);
String8 String8_pushf(MemoryArena *arena, const char *format, ...);

String8 escape_json_string(MemoryArena *arena, String8 input);

b32 String8_Equal(String8 a, String8 b);

String8 *get_command_line_as_utf8(MemoryArena *arena, int *out_argcount );
