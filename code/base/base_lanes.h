/* date = December 6th 2025 3:00 am */

#ifndef LANES_H
#define LANES_H

typedef u64 barrier;

typedef umm thread_handle;

typedef struct thread_context thread_context;
struct thread_context
{
    s64 LaneCount;
    s64 LaneIndex;
    
    thread_handle Handle;
    
    u64 *SharedStorage;
    barrier Barrier;
    
    arena *Arena;
};

internal void ThreadInit(thread_context *ContextToSelect);

#endif //LANES_H
