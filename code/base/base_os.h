/* date = December 6th 2025 2:34 pm */

#ifndef OS_H
#define OS_H

//~ Types
typedef struct entry_point_params entry_point_params;
struct entry_point_params
{
    thread_context Context;
    int ArgsCount;
    char **Args;
};

typedef struct OS_profiler OS_profiler;
struct OS_profiler
{
    s64 Start;
    s64 End;
};

//~ Globals
global_variable u8 LogBuffer[KB(64)];
global_variable OS_profiler GlobalProfiler = {}; 
global_variable char *GlobalProfilerPrefix = ""; 

//~ Functions
#define ENTRY_POINT(Name) void *Name(entry_point_params *Params)
typedef ENTRY_POINT(entry_point_func);

C_LINKAGE ENTRY_POINT(EntryPoint);

#if OS_LINUX
# include "base_os_linux.h"
#endif

internal str8  OS_ReadEntireFileIntoMemory(char *FileName);
internal void  OS_FreeFileMemory(str8 File);
internal b32   OS_WriteEntireFile(char *FileName, str8 File);
internal void  OS_PrintFormat(char *Format, ...);
internal void  OS_BarrierWait(barrier Barrier);
internal void  OS_SetThreadName(str8 ThreadName);
internal void *OS_Allocate(umm Size);
internal void  OS_BarrierWait(barrier Barrier);
internal s64   OS_GetWallClock(void);
internal void  OS_Sleep(u32 MicroSeconds);
internal void  OS_ChangeDirectory(char *Path);
//- OS agnostic, implemented in `base_os.c`.
internal inline f32 OS_SecondsElapsed(s64 Start, s64 End);
internal inline f32 OS_MSElapsed(s64 Start, s64 End);
internal void OS_ProfileInit(char *Prefix);
internal void OS_ProfileAndPrint(char *Label);

#define Log(Format, ...)      OS_PrintFormat((char *)(Format), ##__VA_ARGS__)
// NOTE(luca): Append '\n', because this macro might be redefined into a visual error log.
#define ErrorLog(Format, ...) Log(ERROR_FMT Format "\n", ERROR_ARG, ##__VA_ARGS__)

//- Helpers 
#ifndef RL_PROFILE
# define RL_PROFILE 0
#endif

#endif //OS_H
