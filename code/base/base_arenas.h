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

arena *ArenaAlloc(void);
void *ArenaPush(arena *Arena, umm Size);

#define PushArray(Arena, type, Count) (type *)ArenaPush((Arena), (Count)*(sizeof(type)))
#define PushStruct(Arena, type) PushArray(Arena, type, 1)

#endif //ARENAS_H
