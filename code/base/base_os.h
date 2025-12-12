/* date = December 6th 2025 2:34 pm */

#ifndef OS_H
#define OS_H

global_variable u8 LogBuffer[Kilobytes(64)];

typedef struct entry_point_params entry_point_params;
struct entry_point_params
{
    thread_context Context;
    int ArgsCount;
    char **Args;
};

#define ENTRY_POINT(Name) void *Name(entry_point_params *Params)
typedef ENTRY_POINT(entry_point_func);
ENTRY_POINT(EntryPoint);

str8 OS_ReadEntireFileIntoMemory(char *FileName);
void OS_PrintFormat(char *Format, ...);
void OS_BarrierWait(barrier Barrier);
void OS_SetThreadName(str8 ThreadName);
void* OS_Allocate(umm Size);
void OS_BarrierWait(barrier Barrier);

#endif //OS_H
