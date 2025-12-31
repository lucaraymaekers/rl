#include "base/base.h"
#include "ex_platform.h"

#include "lib/md5.h"

#if OS_LINUX
# include "ex_platform_linux.cpp"
#elif OS_WINDOWS
# include "ex_platform_windows.cpp"
#endif

UPDATE_AND_RENDER(UpdateAndRenderStub) { return false; }

#if 1
#define AppLog(Format, ...) do {if(*Running)Log(Format, ##__VA_ARGS__);} while(0)
#else
#define AppLog(Format, ...) NoOp
#endif

typedef struct app_code app_code;
struct app_code
{
    update_and_render *UpdateAndRender;
    
    b32 Loaded;
    char *LibraryPath;
};

internal void
LinuxLoadAppCode(app_code *Code, app_state *AppState, struct timespec *LastWriteTime, void **Library)
{
    struct stat Stats = {};
    stat(Code->LibraryPath, &Stats);
    umm Size = Stats.st_size;
    struct timespec CurrentWriteTime = Stats.st_mtim;
    b32 WasWritten = !(CurrentWriteTime.tv_sec == LastWriteTime->tv_sec && 
                       CurrentWriteTime.tv_nsec == LastWriteTime->tv_nsec);
    
    if(Size && (!Code->Loaded || WasWritten))
    {
        *LastWriteTime = CurrentWriteTime;
        
        if(*Library)
        {
            dlclose(*Library);
        }
        
        *Library = dlopen(Code->LibraryPath, RTLD_NOW);
        if(*Library)
        {
            // Load code from library
            Code->UpdateAndRender = (update_and_render *)dlsym(*Library, "UpdateAndRender");
            if(Code->UpdateAndRender)
            {
#if RL_INTERNAL
                AppState->Reloaded = true;
#endif
                Code->Loaded = true;
                Log("\nLibrary reloaded.\n");
            }
            else
            {
                Code->Loaded = false;
                ErrorLog("Could not find UpdateAndRender.");
            }
        }
        else
        {
            Code->Loaded = false;
            ErrorLog("%s", dlerror());
        }
    }
    
    if(!Code->Loaded)
    {
        Code->UpdateAndRender = UpdateAndRenderStub;
    }
}

C_LINKAGE ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        arena *PermanentCPUArena = ArenaAlloc(.Size = GB(3));
        arena *CPUFrameArena = ArenaAlloc(.Size = GB(1));
        
        b32 *Running = PushStruct(PermanentCPUArena, b32);
        *Running = true;
        
        app_offscreen_buffer Buffer = {};
        Buffer.Width = 960;
        Buffer.Height = 960;
        Buffer.BytesPerPixel = 4;
        Buffer.Pitch = Buffer.BytesPerPixel*Buffer.Width;
        Buffer.Pixels = PushArray(PermanentCPUArena, u8, Buffer.Pitch*Buffer.Height);
        
        P_context PlatformContext = P_ContextInit(PermanentCPUArena, &Buffer, Running);
        if(!PlatformContext)
        {
            ErrorLog("Could not initialize graphical context, running in headless mode.");
        }
        
        app_state AppState = {};
        AppState.PermanentArena = PermanentCPUArena;
#if RL_INTERNAL
        AppState.DebuggerAttached = GlobalDebuggerIsAttached;
#endif
        
        app_input Input[2] = {};
        app_input *NewInput = &Input[0];
        app_input *OldInput = &Input[1];
        
        s64 LastCounter = OS_GetWallClock();
        s64 FlipWallClock = LastCounter;
        f32 GameUpdateHz = 144.0f;
        f32 TargetSecondsPerFrame = 1.0f/GameUpdateHz; 
        
        app_code Code = {};
        
#if OS_LINUX        
        struct timespec LastWriteTime = {};
        Code.LibraryPath = "./build/app.so";
        void *LibraryHandle = 0;
        LinuxLoadAppCode(&Code, &AppState, &LastWriteTime, &LibraryHandle);
#endif
        
        b32 Paused = false;
        
        while(*Running)
        {
            OS_ProfileInit();
            
            umm CPUBackPos = BeginScratch(CPUFrameArena);
            
            // Prepare  Input
            { 
                *NewInput = *OldInput;
                NewInput->Text.Count = 0;
                for EachIndex(Idx, PlatformButton_Count)
                {
                    NewInput->Buttons[Idx].HalfTransitionCount = 0;
                }
                NewInput->dtForFrame = TargetSecondsPerFrame;
            }
            
            OS_ProfileAndPrint("P_InitSetup");
            
#if OS_LINUX      
            // Load application code
            LinuxLoadAppCode(&Code, &AppState, &LastWriteTime, &LibraryHandle);
            OS_ProfileAndPrint("P_Code");
#endif
            
            P_ProcessMessages(PlatformContext, NewInput, &Buffer, Running);
            
            OS_ProfileAndPrint("P_Messages");
            
            if(CharPressed(NewInput, 'p', PlatformKeyModifier_Alt)) Paused = !Paused;
            
            if(!Paused)
            {
                *Running = *Running &&  !Code.UpdateAndRender(ThreadContext, &AppState, CPUFrameArena, &Buffer, NewInput);
            }
            
            OS_ProfileAndPrint("P_UpdateAndRender");
            
            P_UpdateImage(PlatformContext, &Buffer);
            
            OS_ProfileAndPrint("P_UpdateImage");
            
            s64 WorkCounter = OS_GetWallClock();
            f32 WorkMSPerFrame = OS_MSElapsed(LastCounter, WorkCounter);
            // Sleep
            {            
                f32 SecondsElapsedForFrame = OS_SecondsElapsed(LastCounter, WorkCounter);
                if(SecondsElapsedForFrame < TargetSecondsPerFrame)
                {
                    f32 SleepUS = ((TargetSecondsPerFrame - 0.001f - SecondsElapsedForFrame)*1000000.0f);
                    if(SleepUS > 0)
                    {
                        // TODO(luca): Intrinsic
                        OS_Sleep((u32)SleepUS);
                    }
                    else
                    {
                        // TODO(luca): Logging
                    }
                    
                    f32 TestSecondsElapsedForFrame = OS_SecondsElapsed(LastCounter, OS_GetWallClock());
                    if(TestSecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        // TODO(luca): Log missed sleep
                    }
                    
                    // NOTE(luca): This is to help against sleep granularity.
                    while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        SecondsElapsedForFrame = OS_SecondsElapsed(LastCounter, OS_GetWallClock());
                    }
                }
                else
                {
                    // TODO(luca): Log missed frame rate!
                }
                
                s64 EndCounter = OS_GetWallClock();
                
                LastCounter = EndCounter;
            }
            
#if 1            
            NewInput->Text.Buffer[NewInput->Text.Count].Codepoint = 0;
            f32 FPS = Min(1000.0f/WorkMSPerFrame, GameUpdateHz);
            AppLog("'%c' (%d, %d) 1:%c 2:%c 3:%c", 
                   (u8)NewInput->Text.Buffer[0].Codepoint,
                   NewInput->MouseX, NewInput->MouseY,
                   (NewInput->Buttons[PlatformButton_Left  ].EndedDown ? 'x' : 'o'),
                   (NewInput->Buttons[PlatformButton_Middle].EndedDown ? 'x' : 'o'),
                   (NewInput->Buttons[PlatformButton_Right ].EndedDown ? 'x' : 'o')); 
            
            AppLog(" %.2fms/f %.0fFPS", (f64)WorkMSPerFrame, (f64)FPS);
            AppLog("\n");
#endif
            
#if RL_PROFILE
            // TODO(luca): Sometimes we hit more than 4ms/f
            if(WorkMSPerFrame >= 4000.0f) DebugBreakOnce;
#endif
            
            Swap(OldInput, NewInput);
            
            OS_ProfileAndPrint("P_Sleep");
            
            FlipWallClock = OS_GetWallClock();
            
            EndScratch(CPUFrameArena, CPUBackPos);
        }
    }
    
    return 0;
}