#include <dlfcn.h>

#include "base/base.h"
#include "platform.h"
#include "platform_linux.cpp"

UPDATE_AND_RENDER(UpdateAndRenderStub) {}

C_LINKAGE ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        arena *PermanentCPUArena = ArenaAlloc(.Size = Gigabytes(3));
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
            ErrorLog("Could not initialize X11, running in headless mode.");
        }
        
        void *Library = 0;
        update_and_render *UpdateAndRender = 0;
        
        app_state AppState = {};
        AppState.PermanentCPUArena = PermanentCPUArena;
        
        app_input Input[2] = {};
        app_input *NewInput = &Input[0];
        app_input *OldInput = &Input[1];
        
        s64 LastCounter = P_GetWallClock();
        s64 FlipWallClock = LastCounter;
        f32 GameUpdateHz = 60.0f;
        f32 TargetSecondsPerFrame = 1.0f/GameUpdateHz; 
        
        while(*Running)
        {
            umm CPUBackPos = BeginScratch(CPUFrameArena);
            
            // Prepare  Input
            { 
                NewInput->Text.Count = 0;
                for(EachIndex(Idx, PlatformButton_Count))
                {
                    NewInput->Buttons[Idx].EndedDown = OldInput->Buttons[Idx].EndedDown;
                    NewInput->Buttons[Idx].HalfTransitionCount = 0;
                }
            }
            
            // Load application code
            {            
                if(Library)
                {
                    dlclose(Library);
                }
                Library = dlopen("./build/app.so", RTLD_NOW);
                if(!Library)
                {
                    char *Error = dlerror();
                    ErrorLog("%s", Error);
                    UpdateAndRender = UpdateAndRenderStub;
                }
                else
                {
                    UpdateAndRender = (update_and_render *)dlsym(Library, "UpdateAndRender");
                    if(!UpdateAndRender)
                    {
                        ErrorLog("Could not find UpdateAndRender.");
                        UpdateAndRender = UpdateAndRenderStub;
                    }
                }
                Assert(UpdateAndRender);
            }
            
            P_ProcessMessages(PlatformContext, NewInput, &Buffer, Running);
            
#if 1            
            NewInput->Text.Buffer[NewInput->Text.Count].Codepoint = 0;
            Log(" '%c' (%d, %d) 1:%c 2:%c 3:%c ", 
                (u8)NewInput->Text.Buffer[0].Codepoint,
                NewInput->MouseX, NewInput->MouseY,
                (NewInput->Buttons[PlatformButton_Left  ].EndedDown ? 'x' : 'o'),
                (NewInput->Buttons[PlatformButton_Middle].EndedDown ? 'x' : 'o'),
                (NewInput->Buttons[PlatformButton_Right ].EndedDown ? 'x' : 'o')); 
#endif
            
            UpdateAndRender(ThreadContext, &AppState, CPUFrameArena, &Buffer, NewInput);
            
            // Sleep
            {            
                s64 WorkCounter = P_GetWallClock();
                f32 WorkMSPerFrame = P_MSElapsed(LastCounter, WorkCounter);
                
                f32 SecondsElapsedForFrame = P_SecondsElapsed(LastCounter, WorkCounter);
                if(SecondsElapsedForFrame < TargetSecondsPerFrame)
                {
                    f32 SleepUS = ((TargetSecondsPerFrame - 0.001f - SecondsElapsedForFrame)*1000000.0f);
                    if(SleepUS > 0)
                    {
                        // TODO(luca): Intrinsic
                        usleep((u32)SleepUS);
                    }
                    else
                    {
                        // TODO(luca): Logging
                    }
                    
                    f32 TestSecondsElapsedForFrame = P_SecondsElapsed(LastCounter, P_GetWallClock());
                    if(TestSecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        // TODO(luca): Log missed sleep
                    }
                    
                    // NOTE(luca): This is to help against sleep granularity.
                    while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        SecondsElapsedForFrame = P_SecondsElapsed(LastCounter, P_GetWallClock());
                    }
                }
                else
                {
                    // TODO(luca): Log missed frame rate!
                }
                
                s64 EndCounter = P_GetWallClock();
                
                // Print elapsed time
                {                
                    f32 MSPerFrame = P_MSElapsed(LastCounter, EndCounter);
                    
                    local_persist s32 Counter = 0;
                    s32 MaxCount = (s32)GameUpdateHz/2;
                    
                    local_persist f32 LastMSPerFrame = WorkMSPerFrame;
                    
                    Counter += 1;
                    if(Counter > MaxCount)
                    {
                        LastMSPerFrame = WorkMSPerFrame;
                        Counter -= MaxCount;
                    }
                    
                    f32 FPS = Minimum(1000.0f/LastMSPerFrame, GameUpdateHz);
                    
                    Log("%.2fms/f %.0fFPS", (f64)LastMSPerFrame, (f64)FPS);
                }
                
                
                Log("\n");
                
                LastCounter = EndCounter;
            }
            
            Swap(OldInput, NewInput);
            
            P_UpdateImage(PlatformContext, &Buffer);
            
            FlipWallClock = P_GetWallClock();
            
            EndScratch(CPUFrameArena, CPUBackPos);
        }
    }
    
    return 0;
}