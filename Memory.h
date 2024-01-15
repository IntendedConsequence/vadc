#pragma once
#include "stdint.h"

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
   return arena->base != nullptr;
}

void resetMemoryArena( MemoryArena *arena );
b32 addressIsInsideArena( MemoryArena *arena, void *address );
b32 isPowerOfTwo( size_t number );
size_t getAlignmentOffset( size_t top, size_t alignment );
void *pushSize( MemoryArena *arena, size_t size, size_t alignment );
void *pushSizeZeroed( MemoryArena *arena, size_t size, size_t alignment );
void *resizeAllocationInArena( MemoryArena *arena, void *oldAddress, size_t oldSize, size_t newSize, size_t alignment );
void freeAllocationInArena(MemoryArena *arena, void *address);

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

#define pushStruct(arena, type) (type *) pushSizeZeroed( arena, sizeof(type), alignof(type) )
#define pushArray(arena, count, type) (type *) pushSizeZeroed( arena, sizeof(type) * (count), alignof(type) )
// #define resizeArray(arena, oldAddress, oldCount, newCount)

const char *copyStringToArena( MemoryArena *arena, const char *stringData, size_t stringLength = 0 );
