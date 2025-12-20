#include <dlfcn.h>

#include "base/base.h"
#include "cu.h"
#include "platform.h"
#include "platform_linux.cpp"

PUSH_WARNINGS;
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"
#define HANDMADE_FONT_IMPLEMENTATION
#include "lib/handmade_font.h"
POP_WARNINGS;

UPDATE_AND_RENDER(UpdateAndRenderStub)
{
    
}

C_LINKAGE ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        arena *PermanentCPUArena = ArenaAlloc(.Size = Gigabytes(3));
        
        b32 *Running = PushStruct(PermanentCPUArena, b32);
        *Running = true;
        
        arena *PermanentGPUArena = CU_ArenaAlloc(PermanentCPUArena, .Size = Megabytes(100));
        
        arena *CPUFrameArena = ArenaAlloc();
        
        app_offscreen_buffer Buffer = {};
        Buffer.Width = 1920/2;
        Buffer.Height = 1080/2;
        Buffer.BytesPerPixel = 4;
        Buffer.Pitch = Buffer.BytesPerPixel*Buffer.Width;
        Buffer.Pixels = PushArray(PermanentCPUArena, u8, Buffer.Pitch*Buffer.Height);
        
        cudaDeviceProp Prop;
        CU_GetDeviceProperties(&Prop, 0);
        arena *GPUFrameArena = CU_ArenaAlloc(PermanentCPUArena);
        
        linux_x11_context LinuxContext = LinuxInitX11(&Buffer);
        if(!LinuxContext.Initialized)
        {
            ErrorLog("Could not initialize X11, running in headless mode.");
        }
        
        void *Library = 0;
        update_and_render *UpdateAndRender = 0;
        
        app_state AppState = {};
        AppState.PermanentGPUArena = PermanentGPUArena;
        AppState.PermanentCPUArena = PermanentCPUArena;
        
        app_input Input[2] = {};
        app_input *NewInput = &Input[0];
        app_input *OldInput = &Input[1];
        
        timespec LastCounter = LinuxGetWallClock();
        timespec FlipWallClock = LinuxGetWallClock();
        f32 GameUpdateHz = 60.0f;
        f32 TargetSecondsPerFrame = 1.0f/GameUpdateHz; 
        
        LinuxSetSIGINT(Running);
        
        while(*Running)
        {
            umm CPUBackPos = BeginScratch(CPUFrameArena);
            umm GPUBackPos = BeginScratch(GPUFrameArena);
            
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
            
            LinuxProcessPendingMessages(&LinuxContext, NewInput, &Buffer, Running);
            
#if 1            
            NewInput->Text.Buffer[NewInput->Text.Count].Codepoint = 0;
            OS_PrintFormat(" '%c' (%d, %d) 1:%c 2:%c 3:%c ", 
                           (u8)NewInput->Text.Buffer[0].Codepoint,
                           NewInput->MouseX, NewInput->MouseY,
                           (NewInput->Buttons[PlatformButton_Left  ].EndedDown ? 'x' : 'o'),
                           (NewInput->Buttons[PlatformButton_Middle].EndedDown ? 'x' : 'o'),
                           (NewInput->Buttons[PlatformButton_Right ].EndedDown ? 'x' : 'o')); 
#endif
            
            UpdateAndRender(ThreadContext, &AppState, CPUFrameArena, GPUFrameArena, &Buffer, NewInput);
            
            // Sleep
            {            
                timespec WorkCounter = LinuxGetWallClock();
                f32 WorkMSPerFrame = (f32)((f32)LinuxGetNSecondsElapsed(LastCounter, WorkCounter)/1000000.f);
                
                f32 SecondsElapsedForFrame = LinuxGetSecondsElapsed(LastCounter, WorkCounter);
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
                    
                    f32 TestSecondsElapsedForFrame = (f32)(LinuxGetSecondsElapsed(LastCounter, LinuxGetWallClock()));
                    if(TestSecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        // TODO(luca): Log missed sleep
                    }
                    
                    // NOTE(luca): This is to help against sleep granularity.
                    while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                    {
                        SecondsElapsedForFrame = LinuxGetSecondsElapsed(LastCounter, LinuxGetWallClock());
                    }
                }
                else
                {
                    // TODO(luca): Log missed frame rate!
                }
                
                timespec EndCounter = LinuxGetWallClock();
                f32 MSPerFrame = (f32)((f32)LinuxGetNSecondsElapsed(LastCounter, EndCounter)/1000000.f);
                
                if(AppState.Initialized)
                {
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
                    DrawTextFormat(CPUFrameArena, &Buffer, &AppState.Font, 140.0f, 30.0f+0.0f*14.0f, 0xFF13171F, 
                                   "%.2fms/f %.0fFPS", (f64)LastMSPerFrame, (f64)FPS);
                }
                
                OS_PrintFormat("\n");
                
                LastCounter = EndCounter;
            }
            
            Swap(OldInput, NewInput);
            
            LinuxUpdateImage(&LinuxContext, &Buffer);
            
            FlipWallClock = LinuxGetWallClock();
            
            EndScratch(CPUFrameArena, CPUBackPos);
            EndScratch(GPUFrameArena, GPUBackPos);
        }
    }
    
    return 0;
}
