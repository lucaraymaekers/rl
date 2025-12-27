/* date = December 6th 2025 2:34 pm */

#ifndef OS_H
#define OS_H

global_variable u8 LogBuffer[KB(64)];

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
internal void  OS_FreeFileMemory(str8 File);
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

//- Helpers 
internal inline f32
OS_SecondsElapsed(s64 Start, s64 End)
{
    f32 Result = ((f32)(End - Start)/1000000000.0f);
    return Result;
}

internal inline f32
OS_MSElapsed(s64 Start, s64 End)
{
    f32 Result = ((f32)(End - Start)/1000000.0f);
    return Result;
}

typedef struct OS_profiler OS_profiler;
struct OS_profiler
{
    s64 Start;
    s64 End;
};

internal OS_profiler
OS_ProfileInit()
{
    OS_profiler Result = {0};
    Result.Start = OS_GetWallClock();
    Result.End = Result.Start;
    return Result;
}

internal void
OS_ProfileAndPrint(char *Label, OS_profiler *Profiler)
{
    Profiler->End = OS_GetWallClock();
    Log(" %s: %.4f\n", Label, OS_MSElapsed(Profiler->Start, Profiler->End));
    Profiler->Start = Profiler->End;
}

#ifndef RL_PROFILE
# define RL_PROFILE 0
#endif

#if !RL_PROFILE
# define OS_ProfileAndPrint(Label, Profiler) NoOp
#endif


#endif //OS_H
