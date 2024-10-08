#include "string8.h"

#include <stdio.h> //vsnprintf
#include <string.h> //strlen

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h> //WideCharToMultiByte, MultiByteToWideChar
#include <Shellapi.h> // CommandLineToArgvW, GetCommandLineW

String8 String8FromPointerSize(const s8 *pointer, strSize size)
{
    String8 result = {0};
    result.begin = pointer;
    result.size = size;

    return result;
}

String8 String8FromRange(const s8 *first, const s8 *one_past_last)
{
    String8 result = {0};
    result.begin = first;
    result.size = one_past_last - first;

    return result;
}

String8 String8FromCString(const char *cstring)
{
    return String8FromPointerSize(cstring, (strSize)strlen(cstring));
}

String8 String8ToCString(MemoryArena *arena, String8 source_string)
{
    s8 *cstring = pushSizeZeroed(arena, source_string.size + 1, TEMP_DEFAULT_ALIGNMENT);
    memmove(cstring, source_string.begin, source_string.size);

    return String8FromPointerSize(cstring, source_string.size);
}

String8 String8_pushfv(MemoryArena *arena, const char *format, va_list args)
{
    va_list args_copy;
    va_copy(args_copy, args);
    int buffer_size = 1024;
    char *buffer = pushArray(arena, buffer_size, char);
    int actual_size = vsnprintf(buffer, buffer_size, format, args);
    int actual_size_with_null = actual_size + 1;

    String8 result = {0};
    if (actual_size < buffer_size)
    {
        char *smaller_buffer = (char *)resizeAllocationInArena(arena, buffer, buffer_size, actual_size_with_null, 1);
        result = String8FromPointerSize((const s8 *)smaller_buffer, actual_size);
    }
    else
    {
        char *larger_buffer = (char *)resizeAllocationInArena(arena, buffer, buffer_size, actual_size_with_null, 1);
        int final_size = vsnprintf(larger_buffer, actual_size_with_null, format, args_copy);
        result = String8FromPointerSize((const s8 *)larger_buffer, final_size);
    }
    va_end(args_copy);

    return result;
}

String8 String8_pushf(MemoryArena *arena, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    String8 result = String8_pushfv(arena, format, args);
    va_end(args);

    return result;
}

int String8_ToWidechar(MemoryArena *arena, wchar_t **dest, String8 source)
{
    int buf_char_count_needed = MultiByteToWideChar(
        CP_UTF8,
        0,
        (const char *)source.begin,
        (int)source.size,
        0,
        0
    );

    if (buf_char_count_needed)
    {
        *dest = pushArray(arena, buf_char_count_needed + 1, wchar_t);
        MultiByteToWideChar(
            CP_UTF8,
            0,
            (const char *)source.begin,
            (int)source.size,
            *dest,
            buf_char_count_needed
        );

        (*dest)[buf_char_count_needed] = 0;
    }

    return buf_char_count_needed;
}

String8 Widechar_ToString8(MemoryArena *arena, const wchar_t *str, size_t str_length)
{
    String8 result = {0};

    if (str_length == 0)
    {
        str_length = wcslen(str);
    }

    int buf_char_count_needed = WideCharToMultiByte(
        CP_UTF8,
        0,
        str,
        (int)str_length,
        0,
        0,
        0,
        0
    );

    if (buf_char_count_needed)
    {
        char *dest = pushArray(arena, buf_char_count_needed + 1, char);
        WideCharToMultiByte(
            CP_UTF8,
            0,
            str,
            (int)str_length,
            dest,
            buf_char_count_needed,
            0,
            0
        );

        dest[buf_char_count_needed] = 0;

        result = String8FromPointerSize((const s8 *)dest, buf_char_count_needed);
    }

    return result;

}

String8 escape_json_string(MemoryArena *arena, String8 input)
{
    String8 empty_result = {0};

    // Allocate buffer (worst case: every char is escaped, plus null terminator)
    size_t input_length = (size_t)input.size;

    char *output = pushArray(arena, input_length * 2 + 1, char);
    if (!output) return empty_result;

    const char *in_ptr = (const char *)input.begin;
    const char *in_ptr_end = (const char *)input.begin + input_length;
    char *out_ptr = output;
    while (in_ptr != in_ptr_end) {
        switch (*in_ptr) {
            case '\"': *out_ptr++ = '\\'; *out_ptr++ = '\"'; break;
            case '\\': *out_ptr++ = '\\'; *out_ptr++ = '\\'; break;
            case '\b': *out_ptr++ = '\\'; *out_ptr++ = 'b'; break;
            case '\f': *out_ptr++ = '\\'; *out_ptr++ = 'f'; break;
            case '\n': *out_ptr++ = '\\'; *out_ptr++ = 'n'; break;
            case '\r': *out_ptr++ = '\\'; *out_ptr++ = 'r'; break;
            case '\t': *out_ptr++ = '\\'; *out_ptr++ = 't'; break;
            default: *out_ptr++ = *in_ptr; break;
        }
        in_ptr++;
    }
    *out_ptr = '\0';

    return String8FromRange((const s8 *)output, (const s8 *)out_ptr);
}

b32 String8_Equal(String8 a, String8 b)
{
    if (a.size != b.size)
    {
        return 0;
    }
    else
    {
        return memcmp(a.begin, b.begin, a.size) == 0;
    }
}

String8 *get_command_line_as_utf8(MemoryArena *arena, int *out_argcount )
{
   int argcount = 0;
   wchar_t **arglist = CommandLineToArgvW( GetCommandLineW(), &argcount );

   int u8argc = argcount;
   // char **u8argv = (char **)malloc( sizeof( char * ) * argcount );
   String8 *result = pushArray(arena, argcount, String8);
   for ( int arg_index = 0; arg_index < argcount; ++arg_index )
   {
      result[arg_index] = Widechar_ToString8(arena, arglist[arg_index], 0);
      // Widechar_ToUTF8( &u8argv[arg_index], arglist[arg_index] );
   }

   *out_argcount = u8argc;

   return result;
}
