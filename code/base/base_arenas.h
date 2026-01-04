/* date = December 6th 2025 2:11 pm */

#ifndef ARENAS_H
#define ARENAS_H

typedef struct arena arena;
struct arena
{
    void *Base;
    umm Pos;
    umm Size;
};

typedef struct arena_alloc_params arena_alloc_params;
struct arena_alloc_params
{
    umm DefaultSize;
    umm Size;
};

#define ArenaAllocDefaultSize MB(64)

#define ArenaAlloc(...) ArenaAlloc_((arena_alloc_params){.DefaultSize = ArenaAllocDefaultSize, ##__VA_ARGS__})
internal arena *ArenaAlloc_(arena_alloc_params Params);
internal void  *ArenaPush(arena *Arena, umm Size);
internal umm BeginScratch(arena *Arena);
internal void EndScratch(arena *Arena, umm BackPos);

#define PushArray(Arena, type, Count) (type *)ArenaPush((Arena), (Count)*(sizeof(type)))
#define PushStruct(Arena, type) PushArray(Arena, type, 1)

#endif //ARENAS_H
