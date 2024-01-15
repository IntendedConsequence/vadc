#include "Memory.h"
#include <string.h> // memset, strlen, memmove

void initializeMemoryArena( MemoryArena *arena, u8 *base, size_t size )
{
   arena->base = base;
   arena->size = size;
   arena->previous_used = 0;
   arena->used = 0;
   arena->temporaryMemoryCount = 0;
}

void resetMemoryArena( MemoryArena *arena )
{
   arena->previous_used = 0;
   arena->used = 0;
   arena->temporaryMemoryCount = 0;
}

b32 addressIsInsideArena( MemoryArena *arena, void *address )
{
   u8 *addressChar = (u8*)address;
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

      return address;
   }
   else
   {
      AssertFail( "Requested chunk size exceeds arena capacity!" );

      return nullptr;
   }
}

void *pushSizeZeroed( MemoryArena *arena, size_t size, size_t alignment )
{
   void *result = pushSize( arena, size, alignment );
   if ( result != nullptr )
   {
      memset( result, 0, size );
   }
   return result;
}

void freeAllocationInArena(MemoryArena *arena, void *address)
{
   if ( address != nullptr )
   {
      if ( addressIsInsideArena( arena, address ) )
      {
         // TODO(irwin):
      }
   }
}

void *resizeAllocationInArena( MemoryArena *arena, void *oldAddress, size_t oldSize, size_t newSize, size_t alignment )
{
   u8 *oldAddressChar = (u8*)oldAddress;

   if ( oldAddress == nullptr || oldSize == 0 )
   {
      return pushSizeZeroed( arena, newSize, alignment );

   }
   else if ( addressIsInsideArena( arena, oldAddress ) )
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
               memset( oldAddressChar + oldSize, 0, newSize - oldSize );
            }

            return oldAddress;
         }
         else
         {
            AssertFail( "ERROR: new size exceeds arena capacity!" );

            return nullptr;
         }

         // NOTE(irwin): there are allocations after oldAddress, so we can't extend it.
         // Reallocate and copy instead.
      }
      else
      {
         void *newAddress = pushSizeZeroed( arena, newSize, alignment );
         if ( newAddress != nullptr )
         {
            memmove( newAddress, oldAddress, oldSize < newSize ? oldSize : newSize );
         }
         return newAddress;
      }
   }
   else
   {
      AssertFail( "ERROR: address lies outside the arena chunk!" );

      return nullptr;
   }
}

TemporaryMemory beginTemporaryMemory( MemoryArena *arena )
{
   TemporaryMemory temporaryMemory = {0};
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

   --temporaryMemory.arena->temporaryMemoryCount;
}

const char *copyStringToArena( MemoryArena *arena, const char *stringData, size_t stringLength )
{
   if ( stringLength == 0 )
      stringLength = strlen( stringData );

   void *copied = pushSizeZeroed( arena, stringLength + 1, alignof(char) );
   if ( copied != nullptr )
   {
      memmove( copied, stringData, stringLength );
   }

   return (const char*)copied;
}
