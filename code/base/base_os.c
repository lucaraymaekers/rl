#if OS_LINUX
# include "base_os_linux.c"
#elif OS_WINDOWS
# include "base_os_windows.c"
#else 
# error "Operating system not provided or supported."
#endif

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

internal void
OS_ProfileInit()
{
    GlobalProfiler.Start = OS_GetWallClock();
    GlobalProfiler.End = GlobalProfiler.Start;
}

internal void
OS_ProfileAndPrint(char *Label)
{
    GlobalProfiler.End = OS_GetWallClock();
    Log(" %s: %.4f\n", Label, OS_MSElapsed(GlobalProfiler.Start, GlobalProfiler.End));
    GlobalProfiler.Start = GlobalProfiler.End;
}
