#ifndef CLING_OS_C
#define CLING_OS_C

typedef struct os_command_result os_command_result; 
struct os_command_result
{
    b32 Error;
    
    umm StdoutBytesToRead;
    b32 StdinBytesToWrite;
    umm StderrBytesToRead;
    int Stdout;
    int Stdin;
    int Stderr;
};

internal str8 OS_PathFromExe(u8 *Buffer, char *ExePath, str8 Path);
internal str8 OS_GetFileName(str8 FileName);
internal void OS_RebuildSelf(arena *Arena, int ArgsCount, char *Args[], char *Env[]);
internal os_command_result  OS_RunCommand(str8 Command, char *Env[], b32 Pipe);

# if OS_WINDOWS
#  include "cling_os_windows.c"
# elif OS_LINUX
#  include "cling_os_linux.c"
# endif

#endif // CLING_OS_C

#if CLING_OS_IMPLEMENTATION
# if OS_WINDOWS
#  include "cling_os_windows.c"
# elif OS_LINUX
#  include "cling_os_linux.c"
# endif

internal str8
OS_PathFromExe(u8 *Buffer, char *ExePath, str8 Path)
{
	str8 Result = {0};
    char *FileName = (char *)Buffer;
    
#if OS_LINUX
    u32 SizeOfFileName = (u32)StringLength(ExePath);
    memcpy(FileName, ExePath, SizeOfFileName);
#elif OS_WINDOWS
    u32 SizeOfFileName = GetModuleFileNameA(0, Buffer, sizeof(Buffer));
#endif
    
    str8 ExeDirPath = {0};
    {
        u32 OnePastLastSlash = 0;
        for EachIndex(Idx, SizeOfFileName)
        {
            if(FileName[Idx] == OS_SlashChar)
            {
                OnePastLastSlash = (u32)Idx + 1;
            }
        }
        
        ExeDirPath.Data = (u8 *)FileName;
        ExeDirPath.Size = OnePastLastSlash;
    }
    
    str8 Base = ExeDirPath;
    umm Size = Base.Size + Path.Size + 1;
    
    umm At = Base.Size;
    for EachIndex(Idx, Path.Size)
    {
        FileName[At] = Path.Data[Idx];
        At += 1;
    }
    
    FileName[Size - 1] = 0;
    
    Result.Data = (u8 *)FileName;
    Result.Size = Size;
    
    return Result;
}

internal str8
OS_GetFileName(str8 FileName)
{
    str8 Result = {0};
    
    u32 OnePastLastSlash = 0;
    for EachIndex(Idx, FileName.Size)
    {
        if(FileName.Data[Idx] == OS_SlashChar)
        {
            OnePastLastSlash = (u32)Idx + 1;
        }
    }
    
    Result = S8From(FileName, OnePastLastSlash);
    
    return Result;
}

#endif
