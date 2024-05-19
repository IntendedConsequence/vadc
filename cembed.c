#include "platform.h"

#define MEMORY_IMPLEMENTATION
#include "memory.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {printf("Usage:\n\tcembed <filename>\n"); return 1;}

    const char *fname = argv[1];

    MemoryArena arena = {0};
    u64 arena_size = Megabytes( 16 );
    void *address = VirtualAlloc( 0, arena_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE );
    initializeMemoryArena( &arena, address, arena_size );

    File_Contents data = read_entire_file( &arena, fname );

    if (!data.bytes_count) {
        fprintf(stderr, "Error opening file: %s.\n", fname);
        return 1;
    }

    const int file_size = (int)data.bytes_count;
    u8 *bytes_data = data.contents;

    printf("/* Embedded file: %s */\n", fname);
    printf("static const unsigned char silero_v31_16k_weights[%d] = {\n", file_size);

    for (int byte_index = 0; byte_index < file_size; ++byte_index)
    {
        printf("0x%02x%s",
                bytes_data[byte_index],
                (byte_index == (file_size - 1)) ? "" : ((byte_index+1) % 16 == 0 ? ",\n" : ","));
    }
    printf("\n};\n");

    return 0;
}
