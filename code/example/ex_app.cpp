#include "base/base.h"
#include "ex_platform.h"

//~ Constants

// AA BB GG RR

#define ColorText          0xff87bfcf
#define ColorButtonText    0xFFFBFDFE
#define ColorPoint         0xFF00FFFF
#define ColorCursor        0xFFFF0000
#define ColorCursorPressed ColorPoint
#define ColorButton        0xFF0172AD
#define ColorButtonHovered 0xFF017FC0
#define ColorButtonPressed 0xFF0987C8
#define ColorBackground    0xFF13171F
#define ColorMapBackground 0xFF3A4151

C_LINKAGE 
UPDATE_AND_RENDER(UpdateAndRender)
{
    ThreadContextSelect(Context);
    
#if RL_INTERNAL    
    GlobalDebuggerIsAttached = App->DebuggerAttached;
#endif
    
    u32 Max = 255;
    u32 NumbersCount = 10;
    s32 *Numbers = 0;
    
    if(!App->Initialized)
    {
        RandomSeed(&App->Series, 0);
        
        App->NumbersArena = ArenaAlloc();
        Numbers = PushArray(App->NumbersArena, s32, 0);
        for EachIndex(Idx, NumbersCount)
        {
            u32 Value = (u32)RandomNext(&App->Series) % Max;
            Numbers[Idx] = Value;
        }
        
        App->Initialized = true;
    }
    
    Numbers = PushArray(App->NumbersArena, s32, 0);
    
    if(WasPressed(Input->Buttons[PlatformButton_Left]))
    {    
        for EachIndex(Idx, NumbersCount)
        {
            u32 Value = (u32)RandomNext(&App->Series) & Max;
            Numbers[Idx] = Value;
        }
    }
    
    
    if(WasPressed(Input->Buttons[PlatformButton_Right]))
    {
        for EachIndex(Outer, NumbersCount)
        {            
            // Sort the array
            for EachIndex(Inner, (NumbersCount - 1))
            {
                if(Numbers[Inner] > Numbers[Inner + 1])
                {
                    Swap(Numbers[Inner], Numbers[Inner + 1]);
                }
            }
        }
        
    }
    
    for EachIndex(Y, Buffer->Height)
    {
        s32 ColumnWidth = Buffer->Width/(s32)NumbersCount;
        
        for EachIndex(ColumnIdx, NumbersCount)
        {
            for EachIndex(X, ColumnWidth)
            {
                u32 Gray = Numbers[ColumnIdx];
                
                u32 Color = ((0xFF << 3*8) |
                             ((Gray & 0xFF) << 1*8) |
                             ((Gray & 0xFF) << 2*8) |
                             ((Gray & 0xFF) << 0*8));
                
                u32 *Pixel = (u32 *)(Buffer->Pixels + 
                                     Y*Buffer->Pitch + 
                                     Buffer->BytesPerPixel*(ColumnWidth*ColumnIdx + X));
                *Pixel = Color;
            }
        }
    }
    
    
    
}