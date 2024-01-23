#pragma once
#include <stdint.h>
#include "utils.h"

typedef struct MemoryArena MemoryArena;
struct MemoryArena
{
   u8 *base;
   size_t size;
   size_t previous_used;
   size_t used;
   int temporaryMemoryCount;
};

void initializeMemoryArena( MemoryArena *arena, u8 *base, size_t size );
inline b32 isMemoryArenaInitialized( MemoryArena *arena )
{
   return arena->base != 0;
}

void resetMemoryArena( MemoryArena *arena );
b32 addressIsInsideArena( MemoryArena *arena, void *address );
b32 isPowerOfTwo( size_t number );
size_t getAlignmentOffset( size_t top, size_t alignment );
void *pushSize( MemoryArena *arena, size_t size, size_t alignment );
void *pushSizeZeroed( MemoryArena *arena, size_t size, size_t alignment );
void *resizeAllocationInArena( MemoryArena *arena, void *oldAddress, size_t oldSize, size_t newSize, size_t alignment );
void freeAllocationInArena( MemoryArena *arena, void *address );

typedef struct TemporaryMemory TemporaryMemory;
struct TemporaryMemory
{
   MemoryArena *arena;
   size_t previous_used;
   size_t used;
};

TemporaryMemory beginTemporaryMemory( MemoryArena *arena );
void endTemporaryMemory( TemporaryMemory temporaryMemory );

#ifdef __cplusplus
struct TemporaryMemoryScoped
{
   TemporaryMemoryScoped( MemoryArena *arena )
   {
      temporaryMemory = beginTemporaryMemory( arena );
   }

   ~TemporaryMemoryScoped()
   {
      endTemporaryMemory( temporaryMemory );
   }

   TemporaryMemory temporaryMemory;
};
#endif // __cplusplus

// #define pushStruct(arena, type) (type *) pushSizeZeroed( arena, sizeof(type), alignof(type) )
// #define pushArray(arena, count, type) (type *) pushSizeZeroed( arena, sizeof(type) * (count), alignof(type) )
// TODO(irwin):
#define TEMP_DEFAULT_ALIGNMENT 8
#define pushStruct(arena, type) (type *) pushSizeZeroed( arena, sizeof(type), TEMP_DEFAULT_ALIGNMENT )
#define pushArray(arena, count, type) (type *) pushSizeZeroed( arena, sizeof(type) * (count), TEMP_DEFAULT_ALIGNMENT )
// #define resizeArray(arena, oldAddress, oldCount, newCount)

const char *copyStringToArena( MemoryArena *arena, const char *stringData, size_t stringLength );
MemoryArena *DEBUG_getDebugArena();

#ifdef MEMORY_IMPLEMENTATION
#include <string.h> // memset, strlen, memmove
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

//#pragma comment(lib, "clang_rt.asan-x86_64.lib")

#include <sanitizer/asan_interface.h>

static u8 debug_arena_buffer_2[Megabytes( 16 )];

static MemoryArena DEBUG_debug_arena_2 = { 0 };

MemoryArena *DEBUG_getDebugArena()
{
   if ( DEBUG_debug_arena_2.base )
   {
      // NOTE(irwin): initialized
      return &DEBUG_debug_arena_2;
   } else
   {
      // NOTE(irwin): not initialized
      //__asan_get_shadow_mapping( &shadow_memory_scale, &shadow_memory_offset );
#if 1
      u64 size = Megabytes( 128 );
      void *address = VirtualAlloc( 0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE );
      initializeMemoryArena( &DEBUG_debug_arena_2, address, size );
#else
      initializeMemoryArena( &DEBUG_debug_arena_2, &debug_arena_buffer_2[0], sizeof( debug_arena_buffer_2 ) );
#endif

      return &DEBUG_debug_arena_2;
   }
}

void initializeMemoryArena( MemoryArena *arena, u8 *base, size_t size )
{
   arena->base = base;
   arena->size = size;
   arena->previous_used = 0;
   arena->used = 0;
   arena->temporaryMemoryCount = 0;

   ASAN_POISON_MEMORY_REGION( base, size );
}

void resetMemoryArena( MemoryArena *arena )
{
   arena->previous_used = 0;
   arena->used = 0;
   arena->temporaryMemoryCount = 0;
   ASAN_POISON_MEMORY_REGION( arena->base, arena->size );
}

b32 addressIsInsideArena( MemoryArena *arena, void *address )
{
   u8 *addressChar = (u8 *)address;
   return (arena->base <= addressChar) && (addressChar < arena->base + arena->size);
}

b32 isPowerOfTwo( size_t number )
{
   return (number & (number - 1)) == 0;
}

size_t getAlignmentOffset( size_t top, size_t alignment )
{
   size_t alignmentMask = alignment - 1;
   size_t alignmentOffset = 0;

   if ( top & alignmentMask )
   {
      alignmentOffset = alignment - (top & alignmentMask);
   }

   return alignmentOffset;
}


void *pushSize( MemoryArena *arena, size_t size, size_t alignment )
{
   Assert( isPowerOfTwo( alignment ) );

   size_t top = (size_t)(arena->base + arena->used);
   size_t alignmentOffset = getAlignmentOffset( top, alignment );
   size += alignmentOffset;

   if ( size <= (arena->size - arena->used) )
   {
      void *address = arena->base + arena->used + alignmentOffset;
      arena->previous_used = arena->used + alignmentOffset;
      arena->used += size;

      ASAN_UNPOISON_MEMORY_REGION( address, size );

      return address;
   } else
   {
      AssertFail( "Requested chunk size exceeds arena capacity!" );

      return 0;
   }
}

void *pushSizeZeroed( MemoryArena *arena, size_t size, size_t alignment )
{
   void *result = pushSize( arena, size, alignment );
   if ( result != 0 )
   {
      memset( result, 0, size );
   }
   return result;
}

void freeAllocationInArena( MemoryArena *arena, void *address )
{
   if ( address != 0 )
   {
      if ( addressIsInsideArena( arena, address ) )
      {
         // TODO(irwin):
      }
   }
}

void *resizeAllocationInArena( MemoryArena *arena, void *oldAddress, size_t oldSize, size_t newSize, size_t alignment )
{
   u8 *oldAddressChar = (u8 *)oldAddress;

   if ( oldAddress == 0 || oldSize == 0 )
   {
      return pushSizeZeroed( arena, newSize, alignment );

   } else if ( addressIsInsideArena( arena, oldAddress ) )
   {
      // NOTE(irwin): if oldAddress was the last allocation we can extend it
      if ( arena->base + arena->previous_used == oldAddressChar )
      {
         if ( arena->previous_used + newSize <= arena->size )
         {
            AssertMessage( getAlignmentOffset( (size_t)oldAddress, alignment ) == 0, "Trying to extend previous allocation with different alignment" );

            arena->used = arena->previous_used + newSize;
            if ( oldSize < newSize )
            {
               ASAN_UNPOISON_MEMORY_REGION( oldAddressChar + oldSize, newSize - oldSize );
               memset( oldAddressChar + oldSize, 0, newSize - oldSize );
            } else
            {
               ASAN_POISON_MEMORY_REGION( oldAddressChar + newSize, oldSize - newSize );
            }

            return oldAddress;
         } else
         {
            AssertFail( "ERROR: new size exceeds arena capacity!" );

            return 0;
         }

         // NOTE(irwin): there are allocations after oldAddress, so we can't extend it.
         // Reallocate and copy instead.
      } else
      {
         void *newAddress = pushSizeZeroed( arena, newSize, alignment );
         if ( newAddress != 0 )
         {
            memmove( newAddress, oldAddress, oldSize < newSize ? oldSize : newSize );
         }
         return newAddress;
      }
   } else
   {
      AssertFail( "ERROR: address lies outside the arena chunk!" );

      return 0;
   }
}

TemporaryMemory beginTemporaryMemory( MemoryArena *arena )
{
   TemporaryMemory temporaryMemory = { 0 };
   temporaryMemory.arena = arena;
   temporaryMemory.previous_used = arena->previous_used;
   temporaryMemory.used = arena->used;

   ++arena->temporaryMemoryCount;

   return temporaryMemory;
}

void endTemporaryMemory( TemporaryMemory temporaryMemory )
{
   Assert( temporaryMemory.arena->temporaryMemoryCount > 0 );
   temporaryMemory.arena->previous_used = temporaryMemory.previous_used;
   temporaryMemory.arena->used = temporaryMemory.used;

   ASAN_POISON_MEMORY_REGION( temporaryMemory.arena->base + temporaryMemory.arena->used, temporaryMemory.arena->size - temporaryMemory.arena->used );

   --temporaryMemory.arena->temporaryMemoryCount;
}

const char *copyStringToArena( MemoryArena *arena, const char *stringData, size_t stringLength )
{
   if ( stringLength == 0 )
      stringLength = strlen( stringData );

   void *copied = pushSizeZeroed( arena, stringLength + 1, TEMP_DEFAULT_ALIGNMENT );
   if ( copied != 0 )
   {
      memmove( copied, stringData, stringLength );
   }

   return (const char *)copied;
}

#endif // MEMORY_IMPLEMENTATION