global_variable s64 GlobalPerfCountFrequency;

//- Helpers 

internal inline u32
SafeTruncateU64(u64 Value)
{
    Assert(Value <= (u64)~0);
    u32 Result = (u32)Value;
    return(Result);
}

//- API 
internal str8 
OS_ReadEntireFileIntoMemory(char *FileName)
{
    str8 Result = {};
    
    HANDLE FileHandle = CreateFileA(FileName, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if(FileHandle != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER FileSize;
        if(GetFileSizeEx(FileHandle, &FileSize))
        {
            u32 FileSize32 = (u32)(FileSize.QuadPart);
            Result.Data = (u8 *)VirtualAlloc(0, FileSize32, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
            if(Result.Data)
            {
                DWORD BytesRead;
                if(ReadFile(FileHandle, Result.Data, FileSize32, &BytesRead, 0) &&
                   (FileSize32 == BytesRead))
                {
                    // NOTE(casey): File read successfully
                    Result.Size = FileSize32;
                }
                else
                {                    
                    ErrorLog("Could not read correctly from '%s'.\n", FileName);
                    
                    if(Result.Data)
                    {
                        VirtualFree(Result.Data, 0, MEM_RELEASE);
                    }
                    
                    Result.Data = 0;
                }
            }
            else
            {
                ErrorLog("Could not allocate memory for reading '%s'.\n", FileName);
            }
        }
        else
        {
            ErrorLog("Could not open '%s' for reading.\n");
        }
    }
    
    return Result;
}

internal b32
OS_WriteEntireFile(char *FileName, str8 File)
{
    b32 Result = false;
    
    HANDLE FileHandle = CreateFileA(FileName, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
    if(FileHandle != INVALID_HANDLE_VALUE)
    {
        DWORD BytesWritten;
        if(WriteFile(FileHandle, File.Data, (DWORD)File.Size, &BytesWritten, 0))
        {
            // NOTE(casey): File read successfully
            Result = (BytesWritten == File.Size);
        }
        else
        {
            ErrorLog("Could not write to '%s'.", FileName);
        }
        
        CloseHandle(FileHandle);
    }
    else
    {
        ErrorLog("Could not open '%s' for writing.", FileName);
    }
    
    return Result;
}

internal void 
OS_PrintFormat(char *Format, ...)
{
    va_list Args;
    va_start(Args, Format);
    
#if 0    
    vsprintf((char *)LogBuffer, Format, Args);
    OutputDebugStringA((char *)LogBuffer);
#else
    vprintf(Format, Args);
#endif
}

internal void
OS_BarrierWait(barrier Barrier)
{
    
}

internal void 
OS_SetThreadName(str8 ThreadName)
{
    
}

internal void *
OS_Allocate(umm Size)
{
    void *Result = VirtualAlloc(0, Size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    return Result;
}

internal s64
OS_GetWallClock(void)
{
    s64 Result = 0;
    
    LARGE_INTEGER Time;
    QueryPerformanceCounter(&Time);
    
    f32 Seconds = (f32)Time.QuadPart/(f32)GlobalPerfCountFrequency;
    // NOTE(luca): Convert to nanoseconds
    Result = (s64)(Seconds*1000.0f*1000.0f*1000.0f);
    
    return Result;
}

internal void
OS_Sleep(u32 Seconds)
{
    //Sleep(Seconds/1000);
}

//~ Entrypoint
int CALLBACK
WinMain(HINSTANCE Instance,
        HINSTANCE PrevInstance,
        LPSTR CommandLine,
        int ShowCode)
{
    u64 SharedStorage = 0;
    entry_point_params Params = {0};
    
    GlobalDebuggerIsAttached = raddbg_is_attached();
    
    Trap();
    
    AllocConsole();
    FILE* stream;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
    
    LARGE_INTEGER PerfCountFrequencyResult;
    QueryPerformanceFrequency(&PerfCountFrequencyResult);
    GlobalPerfCountFrequency = PerfCountFrequencyResult.QuadPart;
    
    thread_context *Context = &Params.Context;
    Context->LaneCount = 1;
    Context->SharedStorage = &SharedStorage;
    ThreadContext = Context;
    
    EntryPoint(&Params);
    return 0;
}