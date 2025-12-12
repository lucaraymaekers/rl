#include "base_arenas.h"

#define MemoryCopy memcpy
#define AtomicAddEvalU64(Pointer, Value) \
(__sync_fetch_and_add((Pointer), (Value), __ATOMIC_SEQ_CST) + (Value));

thread_static thread_context *ThreadContext;

#define LaneCount() (ThreadContext->LaneCount)
#define LaneIndex() (ThreadContext->LaneIndex)

void ThreadContextSelect(thread_context *Context)
{
    ThreadContext = Context;
}

void LaneIceberg(void)
{
    OS_BarrierWait(ThreadContext->Barrier);
}

void LaneSyncU64(u64 *Value, s64 SourceIndex)
{
    if(LaneIndex() == SourceIndex)
    {
        MemoryCopy(ThreadContext->SharedStorage, Value, sizeof(u64));
    }
    LaneIceberg();
    
    if(LaneIndex() != SourceIndex)
    {
        MemoryCopy(Value, ThreadContext->SharedStorage, sizeof(u64));
    }
    LaneIceberg();
}

range_s64 LaneRange(s64 ValuesCount)
{
    range_s64 Result = {0};
    
    s64 ValuesPerThread = ValuesCount/LaneCount();
    
    s64 LeftoverValuesCount = ValuesCount%LaneCount();
    b32 ThreadHasLeftover = (LaneIndex() < LeftoverValuesCount);
    s64 LeftoversBeforeThisThreadIndex = ((ThreadHasLeftover) ? 
                                          LaneIndex(): 
                                          LeftoverValuesCount);
    
    Result.Min = (ValuesPerThread*LaneIndex()+
                  LeftoversBeforeThisThreadIndex);
    Result.Max = (Result.Min + ValuesPerThread + !!ThreadHasLeftover);
    
    return Result;
}

void ThreadInit(thread_context *ContextToSelect)
{
    ThreadContextSelect(ContextToSelect);
    
    ThreadContext->Arena = ArenaAlloc();
    
    u8 ThreadNameBuffer[16] = {0};
    str8 ThreadName = {0};
    ThreadName.Data = ThreadNameBuffer;
    ThreadName.Size = 1;
    ThreadName.Data[0] = (u8)LaneIndex() + '0';
    OS_SetThreadName(ThreadName);
}