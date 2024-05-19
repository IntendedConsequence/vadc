#pragma once
#include "utils.h"
#include "memory.h"

typedef struct File_Contents File_Contents;

struct File_Contents
{
   u8 *contents;
   u64 bytes_count;
};

static inline File_Contents read_entire_file(MemoryArena *arena, const char *path)
{
   File_Contents result = {0};

   FILE *fp = fopen( path, "rb" );
   if ( !fp )
   {
      return result;
   }

   fseek( fp, 0, SEEK_END );
   const int fsize = ftell( fp );

   fseek( fp, 0, SEEK_SET );
   
   u8 *b = pushArray(arena, fsize, u8);

   fread( b, fsize, 1, fp );
   fclose( fp );

   result.contents = b;
   result.bytes_count = fsize;

   return result;
}


