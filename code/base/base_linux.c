// Standard
#include <stdio.h>
#include <string.h>

// Linux
#include <pthread.h>
#include <linux/prctl.h> 
#include <sys/prctl.h>

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>

PUSH_WARNINGS
#define STB_SPRINTF_IMPLEMENTATION
#include "lib/stb_sprintf.h"
POP_WARNINGS

#include "base_arenas.h"

typedef void *pthread_entry_point_func(void *);

//~ Types
typedef struct os_thread os_thread;
struct os_thread
{
    pthread_t Handle;
    void *Result;
    
    thread_context Context;
    entry_point_params Params;
};

//~ Syscalls

//- Debug utilities 
void AssertErrno(b32 Expression)
{
    if(!Expression)
    {
        int Errno = errno;
        char *Error = strerror(Errno);
        Assert(0);
    }
}

str8 OS_ReadEntireFileIntoMemory(char *FileName)
{
    str8 Result = {};
    
    if(FileName)
    {
        int File = open(FileName, O_RDONLY);
        
        if(File != -1)
        {
            struct stat StatBuffer = {};
            int Error = fstat(File, &StatBuffer);
            AssertErrno(Error != -1);
            
            Result.Size = StatBuffer.st_size;
            Result.Data = (u8 *)mmap(0, Result.Size, PROT_READ, MAP_PRIVATE, File, 0);
            AssertErrno(Result.Data != MAP_FAILED);
        }
    }
    
    return Result;
}

void OS_PrintFormat(char *Format, ...)
{
    va_list Args;
    va_start(Args, Format);
    
    vprintf(Format, Args);
    
    // TODO(luca): Make these thread-safe.
#if 0    
    int Length = stbsp_vsprintf((char *)LogBuffer, Format, Args);
    smm BytesWritten = write(STDOUT_FILENO, LogBuffer, Length);
    AssertErrno(BytesWritten == Length);
#endif
    
}

//~ Threads
void OS_BarrierWait(barrier Barrier)
{
    s32 Ret = pthread_barrier_wait((pthread_barrier_t *)Barrier);
    
    AssertErrno(Ret == 0 || Ret == PTHREAD_BARRIER_SERIAL_THREAD);
}

void OS_SetThreadName(str8 ThreadName)
{
    Assert(ThreadName.Size <= 16 -1);
    s32 Ret = prctl(PR_SET_NAME, ThreadName.Data);
    AssertErrno(Ret != -1);
}

void* OS_Allocate(umm Size)
{
    void *Result = mmap(0, Size, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
    return Result;
}

ENTRY_POINT(ThreadInitEntryPoint)
{
    ThreadInit(&Params->Context);
    return EntryPoint(Params);
}

//~ Entrypoint
void LinuxMainEntryPoint(int ArgsCount, char **Args)
{
    arena *Arena = ArenaAlloc();
    
    char ThreadName[16] = "Main";
    
#if FORCE_THREADS_COUNT
    s64 ThreadsCount = FORCE_THREADS_COUNT;
#else
    s64 ThreadsCount = get_nprocs();
#endif
    
    os_thread *Threads = PushArray(Arena, os_thread, ThreadsCount);
    s32 Ret = 0;
    
    prctl(PR_SET_NAME, ThreadName);
    
    u64 SharedStorage = 0;
    
    barrier Barrier = (u64)ArenaPush(Arena, 1);
    
    pthread_barrier_init((pthread_barrier_t *)Barrier, 0, (u32)ThreadsCount);
    
    for(s64 Index = 0; Index < ThreadsCount; Index += 1)
    {
        entry_point_params *Params = &Threads[Index].Params;
        Params->Context.LaneIndex = Index;
        Params->Context.LaneCount = ThreadsCount;
        Params->Context.Barrier   = Barrier;
        Params->Context.SharedStorage = &SharedStorage;
        Params->Args = Args;
        Params->ArgsCount = ArgsCount;
        
        Ret = pthread_create(&Threads[Index].Handle, 0, (pthread_entry_point_func *)ThreadInitEntryPoint, Params);
        Assert(Ret == 0);
    }
    
    for(s64 Index = 0; Index < ThreadsCount; Index += 1)
    {
        pthread_join(Threads[Index].Handle, &Threads[Index].Result);
    }
}

int main(int ArgsCount, char **Args)
{
    LinuxMainEntryPoint(ArgsCount, Args);
    return 0;
}