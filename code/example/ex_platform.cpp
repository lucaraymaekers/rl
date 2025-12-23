#include "base/base.h"
#include "ex_platform.h"

#if OS_LINUX
# include "ex_platform_linux.cpp"
#elif OS_WINDOWS
# include "ex_platform_windows.cpp"
#endif

UPDATE_AND_RENDER(UpdateAndRenderStub) {}

#define AppLog(Format, ...) do {if(*Running)Log(Format, ##__VA_ARGS__);} while(0)

C_LINKAGE ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        //Trap();
        arena *PermanentCPUArena = ArenaAlloc(.Size = GB(3));
        arena *CPUFrameArena = ArenaAlloc();
        
        b32 *Running = PushStruct(PermanentCPUArena, b32);
        *Running = true;
        
        app_offscreen_buffer Buffer = {};
        Buffer.Width = 1920/2;
        Buffer.Height = 1080/2;
        Buffer.BytesPerPixel = 4;
        Buffer.Pitch = Buffer.BytesPerPixel*Buffer.Width;
        Buffer.Pixels = PushArray(PermanentCPUArena, u8, Buffer.Pitch*Buffer.Height);
        
        P_context PlatformContext = P_ContextInit(PermanentCPUArena, &Buffer, Running);
        if(!PlatformContext)
        {
            ErrorLog("Could not initialize graphical context, running in headless mode.");
        }
        
        void *Library = 0;
        update_and_render *UpdateAndRender = UpdateAndRenderStub;
        
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
        
#if OS_LINUX        
        char *LibraryFilePath = "./build/app.so";
        struct timespec LastWriteTime = {};;
#endif
        
        while(*Running)
        {
            umm CPUBackPos = BeginScratch(CPUFrameArena);
            
            // Prepare  Input
            { 
                *NewInput = *OldInput;
                NewInput->Text.Count = 0;
                for EachIndex(Idx, PlatformButton_Count)
                {
                    NewInput->Buttons[Idx].HalfTransitionCount = 0;
                }
            }
            
#if OS_LINUX      
            // Load application code
            {
                struct stat LibraryFileStats = {};
                stat(LibraryFilePath, &LibraryFileStats);
                
                if(LibraryFileStats.st_size && 
                   !(LibraryFileStats.st_mtim.tv_sec == LastWriteTime.tv_sec &&
                     LibraryFileStats.st_mtim.tv_nsec == LastWriteTime.tv_nsec))
                {
                    if(Library)
                    {
                        dlclose(Library);
                    }
                    
                    Library = dlopen(LibraryFilePath, RTLD_NOW);
                    if(!Library)
                    {
                        ErrorLog("%s", dlerror());
                        UpdateAndRender = UpdateAndRenderStub;
                    }
                    else
                    {
                        LastWriteTime = LibraryFileStats.st_mtim;
                        // Load code from library
                        UpdateAndRender = (update_and_render *)dlsym(Library, "UpdateAndRender");
                        if(UpdateAndRender)
                        {
                            Log("\nLibrary reloaded.\n");
                        }
                        else
                        {
                            ErrorLog("Could not find UpdateAndRender.");
                            UpdateAndRender = UpdateAndRenderStub;
                        }
                    }
                }
            }
            
            Assert(UpdateAndRender);
#endif
            
            P_ProcessMessages(PlatformContext, NewInput, &Buffer, Running);
            
#if 1            
            NewInput->Text.Buffer[NewInput->Text.Count].Codepoint = 0;
            AppLog(" '%c' (%d, %d) 1:%c 2:%c 3:%c ", 
                   (u8)NewInput->Text.Buffer[0].Codepoint,
                   NewInput->MouseX, NewInput->MouseY,
                   (NewInput->Buttons[PlatformButton_Left  ].EndedDown ? 'x' : 'o'),
                   (NewInput->Buttons[PlatformButton_Middle].EndedDown ? 'x' : 'o'),
                   (NewInput->Buttons[PlatformButton_Right ].EndedDown ? 'x' : 'o')); 
#endif
            
            UpdateAndRender(ThreadContext, &AppState, CPUFrameArena, &Buffer, NewInput);
            
            // Sleep
            {            
                s64 WorkCounter = OS_GetWallClock();
                f32 WorkMSPerFrame = P_MSElapsed(LastCounter, WorkCounter);
                
                f32 SecondsElapsedForFrame = P_SecondsElapsed(LastCounter, WorkCounter);
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
                    
                    f32 TestSecondsElapsedForFrame = P_SecondsElapsed(LastCounter, OS_GetWallClock());
                    if(TestSecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        // TODO(luca): Log missed sleep
                    }
                    
                    // NOTE(luca): This is to help against sleep granularity.
                    while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        SecondsElapsedForFrame = P_SecondsElapsed(LastCounter, OS_GetWallClock());
                    }
                }
                else
                {
                    // TODO(luca): Log missed frame rate!
                }
                
                s64 EndCounter = OS_GetWallClock();
                
                // Print elapsed time
                {                
                    f32 MSPerFrame = P_MSElapsed(LastCounter, EndCounter);
                    
                    // TODO(luca): Print a recent average instead.
                    
#if 0                    
                    local_persist s32 UpdateCounter = 0;
                    s32 MaxCount = (s32)GameUpdateHz/2;
                    
                    local_persist f32 LastMSPerFrame = WorkMSPerFrame;
                    
                    UpdateCounter += 1;
                    if(UpdateCounter > MaxCount)
                    {
                        LastMSPerFrame = WorkMSPerFrame;
                        UpdateCounter -= MaxCount;
                    }
#else
                    f32 LastMSPerFrame = WorkMSPerFrame;
#endif
                    
                    f32 FPS = Min(1000.0f/LastMSPerFrame, GameUpdateHz);
                    
                    AppLog("%.2fms/f %.0fFPS", (f64)LastMSPerFrame, (f64)FPS);
                }
                
                AppLog("\n");
                
                LastCounter = EndCounter;
            }
            
            Swap(OldInput, NewInput);
            
            P_UpdateImage(PlatformContext, &Buffer);
            
            FlipWallClock = OS_GetWallClock();
            
            EndScratch(CPUFrameArena, CPUBackPos);
        }
    }
    
    return 0;
}