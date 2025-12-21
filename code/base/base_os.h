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

C_LINKAGE ENTRY_POINT(EntryPoint);

internal str8  OS_ReadEntireFileIntoMemory(char *FileName);
internal b32   OS_WriteEntireFile(char *FileName, str8 File);
internal void  OS_PrintFormat(char *Format, ...);
internal void  OS_BarrierWait(barrier Barrier);
internal void  OS_SetThreadName(str8 ThreadName);
internal void *OS_Allocate(umm Size);
internal void  OS_BarrierWait(barrier Barrier);
internal s64   OS_GetWallClock(void);
internal void  OS_Sleep(u32 MicroSeconds);


#define Log(Format, ...)      OS_PrintFormat((char *)(Format), ##__VA_ARGS__)
// NOTE(luca): Append '\n', because this macro might be redefined into a visual error log.
#define ErrorLog(Format, ...) Log(ERROR_FMT Format "\n", ERROR_ARG, ##__VA_ARGS__) 

#endif //OS_H
