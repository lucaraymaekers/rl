#include "base/base.h"
#include "ex_platform.h"

#include "lib/md5.h"

#if OS_LINUX
# include "ex_platform_linux.cpp"
#elif OS_WINDOWS
# include "ex_platform_windows.cpp"
#endif

C_LINKAGE ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        arena *PermanentArena = ArenaAlloc(.Size = GB(3));
        arena *FrameArena = ArenaAlloc(.Size = GB(1));
        
        b32 *Running = PushStruct(PermanentArena, b32);
        *Running = true;
        
        app_offscreen_buffer Buffer = {};
        Buffer.Width = 960;
        Buffer.Height = 960;
        Buffer.BytesPerPixel = 4;
        Buffer.Pitch = Buffer.BytesPerPixel*Buffer.Width;
        Buffer.Pixels = PushArray(PermanentArena, u8, Buffer.Pitch*Buffer.Height);
        
        P_context PlatformContext = P_ContextInit(PermanentArena, &Buffer, Running);
        if(!PlatformContext)
        {
            ErrorLog("Could not initialize graphical context, running in headless mode.");
        }
        
        str8 ExeDirPath = {};
        {        
            u32 OnePastLastSlash = 0;
#if OS_LINUX
            char *FileName = Params->Args[0];
            u32 SizeOfFileName = (u32)StringLength(FileName);
#elif OS_WINDOWS
            char *FileName = PushArray(PermanentArena, char, 1024);
            u32 SizeOfFileName = GetModuleFileNameA(0, FileName, 1024);
            Win32LogIfError();
#else
# error "OS not supported" 
#endif
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
        
        app_state AppState = {};
        AppState.ExeDirPath = ExeDirPath;
        AppState.PermanentArena = PermanentArena;
        
#if RL_INTERNAL
        AppState.DebuggerAttached = GlobalDebuggerIsAttached;
#endif
        
        app_input Input[2] = {};
        app_input *NewInput = &Input[0];
        app_input *OldInput = &Input[1];
        
        s64 LastCounter = OS_GetWallClock();
        s64 FlipWallClock = LastCounter;
#if EX_FORCE_UPDATE_HZ
        f32 GameUpdateHz = EX_FORCE_UPDATE_HZ;
#else
        f32 GameUpdateHz = 144.0f;
#endif
        f32 TargetSecondsPerFrame = 1.0f/GameUpdateHz; 
        
        app_code Code = {};
        
        s64 LastWriteTime = {};
#if OS_LINUX        
        Code.LibraryPath = PathFromExe(PermanentArena, &AppState, S8("ex_app.so"));
        void *LinuxLibraryHandle = 0;
        Code.LibraryHandle = (umm)LinuxLibraryHandle;
#else
        Code.LibraryPath = PathFromExe(PermanentArena, &AppState, S8("ex_app.dll"));
        HMODULE Win32LibraryHandle = 0;
        Code.LibraryHandle = (umm)Win32LibraryHandle;
#endif
        
        b32 Paused = false;
        
        while(*Running)
        {
            OS_ProfileInit("P");
            
            umm CPUBackPos = BeginScratch(FrameArena);
            
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
            
            OS_ProfileAndPrint("InitSetup");
            
            // Load application code
            P_LoadAppCode(FrameArena, &Code, &AppState, &LastWriteTime);
            OS_ProfileAndPrint("Code");
            
            P_ProcessMessages(PlatformContext, NewInput, &Buffer, Running);
            
            OS_ProfileAndPrint("Messages");
            
            if(CharPressed(NewInput, 'p', PlatformKeyModifier_Alt)) Paused = !Paused;
            
            if(!Paused)
            {
                b32 ShouldQuit = Code.UpdateAndRender(ThreadContext, &AppState, FrameArena, &Buffer, NewInput);
                // NOTE(luca): Since UpdateAndRender can take some time, there could have been a signal sent to INT the app.
                ReadWriteBarrier;
                *Running = *Running && !ShouldQuit;
            }
            
            OS_ProfileAndPrint("UpdateAndRender");
            
            P_UpdateImage(PlatformContext, &Buffer);
            
            OS_ProfileAndPrint("UpdateImage");
            
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
            
            NewInput->Text.Buffer[NewInput->Text.Count].Codepoint = 0;
            
#if RL_PROFILE
            // TODO(luca): Sometimes we hit more than 4ms/f
            if(WorkMSPerFrame >= 4000.0f) DebugBreakOnce;
#endif
            
            Swap(OldInput, NewInput);
            
            OS_ProfileAndPrint("Sleep");
            
            f32 FPS = Min(1000.0f/WorkMSPerFrame, GameUpdateHz);
            Log("'%c' (%d, %d) 1:%c 2:%c 3:%c", 
                (u8)NewInput->Text.Buffer[0].Codepoint,
                NewInput->MouseX, NewInput->MouseY,
                (NewInput->Buttons[PlatformButton_Left  ].EndedDown ? 'x' : 'o'),
                (NewInput->Buttons[PlatformButton_Middle].EndedDown ? 'x' : 'o'),
                (NewInput->Buttons[PlatformButton_Right ].EndedDown ? 'x' : 'o')); 
            
            Log(" %.2fms/f %.0fFPS", (f64)WorkMSPerFrame, (f64)FPS);
            Log("\n");
            
            FlipWallClock = OS_GetWallClock();
            
            EndScratch(FrameArena, CPUBackPos);
        }
    }
    
    return 0;
}
