#define BASE_NO_ENTRYPOINT 1
#include "base/base.h"
#include "ex_platform.h"

#include <math.h>

#include "base/base.c"

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

internal inline u32 *
PixelFromBuffer(app_offscreen_buffer *Buffer, u32 X, u32 Y)
{
    Assert(X >= 0 && X < Buffer->Width);
    Assert(Y >= 0 && Y < Buffer->Height);
    
    u32 *Pixel = (u32 *)(Buffer->Pixels + Y*Buffer->Pitch + X*Buffer->BytesPerPixel);
    return Pixel;
}

internal void 
BubbleSort(u32 Count, u32 *List)
{
    for EachIndex(Outer, Count)
    {            
        b32 IsArraySorted = true;
        for EachIndex(Inner, (Count - 1))
        {
            if(List[Inner] > List[Inner + 1])
            {
                Swap(List[Inner], List[Inner + 1]);
                IsArraySorted = false;
            }
        }
        if(IsArraySorted)
        {
            break;
        }
    }
}

internal void 
MergeSortRecursive(u32 Count, u32 *First, u32 *Out)
{
    if(Count == 1)
    {
        // Nothing to be done.
    }
    else if(Count == 2)
    {
        if(First[0] > First[1]) 
        {
            Swap(First[0], First[1]);
        }
    }
    else
    {
        u32 Half0 = Count/2;
        u32 Half1 = Count - Half0;
        
        u32 *InHalf0 = First;
        u32 *InHalf1 = First + Half0;
        
        MergeSortRecursive(Half0, InHalf0, Out);
        MergeSortRecursive(Half1, InHalf1, Out + Half0);
        
        u32 *Write = Out;
        u32 *ReadHalf0 = InHalf0;
        u32 *ReadHalf1 = InHalf1;
        u32 *WriteEnd = Write + Count;
        u32 *End = First + Count;
        
        while(Write != WriteEnd)
        {
            if(ReadHalf1 == End)
            {
                *Write++ = *ReadHalf0;
            }
            else if(ReadHalf0 == InHalf1)
            {
                *Write++ = *ReadHalf1++;
            }
            else if(*ReadHalf0 < *ReadHalf1)
            {
                *Write++ = *ReadHalf0++;
            }
            else
            {
                *Write++ = *ReadHalf1++;
            }
        }
        
        for EachIndex(Idx, Count)
        {
            Swap(First[Idx], Out[Idx]);
        }
    }
}

internal void 
MergeSortIterative(u32 Count, u32 *In, u32 *Temp)
{
    u32 PairSize = 2;
    
    if((s32)(ceilf(log2f((f32)Count))) & 1)
    {
        MemoryCopy(Temp, In, sizeof(u32)*Count);
        Swap(In, Temp);
    }
    
    b32 Toggle = false;
    b32 LastMerge = false;
    while(!LastMerge)
    {    
        u32 PairsCount = CeilIntegerDiv(Count, PairSize);
        u32 Sorted = Count;
        
        LastMerge = (PairSize > Count);
        
        for EachIndex(Idx, PairsCount)
        {
            u32 *Group = In + Idx*PairSize;
            u32 *Out = Temp + Idx*PairSize;
            
            u32 GroupSize = ((PairSize <= Sorted) ? PairSize: Sorted);
            Sorted -= PairSize;
            u32 Half = PairSize/2;
            
            if(Half > GroupSize)
            {
                // NOTE(luca): Leftover group, just copy the values over
                Half = GroupSize;
            }
            
            u32 Half0 = Half;
            u32 Half1 = GroupSize - Half0;
            u32 *InHalf0 = Group;
            u32 *InHalf1 = Group + Half0;
            
            u32 *ReadHalf0 = InHalf0;
            u32 *ReadHalf1 = InHalf1;
            u32 *ReadEnd = InHalf0 + GroupSize;
            u32 *Write = Out;
            u32 *WriteEnd = Out + GroupSize;
            
            while(Write != WriteEnd)
            {
                if(ReadHalf0 == InHalf1)
                {
                    *Write++ = *ReadHalf1++;
                }
                else if(ReadHalf1 == ReadEnd)
                {
                    *Write++ = *ReadHalf0++;
                }
                else if(*ReadHalf0 < *ReadHalf1)
                {
                    *Write++ = *ReadHalf0++;
                }
                else
                {
                    *Write++ = *ReadHalf1++;
                }
            }
        }
        
        PairSize *= 2;
        
        Toggle = !Toggle;
        
        Swap(Temp, In);
    }
}

C_LINKAGE 
UPDATE_AND_RENDER(UpdateAndRender)
{
    ThreadContextSelect(Context);
    
#if RL_INTERNAL    
    GlobalDebuggerIsAttached = App->DebuggerAttached;
#endif
    
    u32 Max = 255;
    u32 NumbersCount = 10000;
    u32 *Numbers = 0;
    
    if(!App->Initialized)
    {
        RandomSeed(&App->Series, 0);
        
        App->NumbersArena = ArenaAlloc();
        Numbers = PushArray(App->NumbersArena, u32, 0);
        for EachIndex(Idx, NumbersCount)
        {
            u32 Value = (u32)RandomNext(&App->Series) & Max;
            Numbers[Idx] = Value;
        }
        
        App->Initialized = true;
    }
    
    Numbers = PushArray(App->NumbersArena, u32, 0);
    
    if(WasPressed(Input->Buttons[PlatformButton_Left]))
    {    
        for EachIndex(Idx, NumbersCount)
        {
            u32 Value = (u32)RandomNext(&App->Series) & (Max);
            Numbers[Idx] = Value;
        }
    }
    
    local_persist b32 SelectedSort = 1;
    if(CharPressed(Input, 's', PlatformKeyModifier_Any))
    {
        SelectedSort = !SelectedSort;
    }
    
    if(WasPressed(Input->Buttons[PlatformButton_Right]))
    {
        if(SelectedSort)
        {
            BubbleSort(NumbersCount, Numbers);
        }
        else
        {            
            u32 *Temp = PushArray(FrameArena, u32, NumbersCount);
            MergeSortIterative(NumbersCount, Numbers, Temp);
        }
        
#if RL_SLOW
        for EachIndex(Idx, NumbersCount - 1)
        {
            Assert(Numbers[Idx] <= Numbers[Idx + 1]);
        }
#endif
    }
    
    for EachIndex(Y, Buffer->Height)
    {
        for EachIndex(X, Buffer->Width)
        {
            u32 Idx = 0;
            if(Buffer->Width > NumbersCount)
            {
                Idx = (u32)(floorf((f32)X/((f32)Buffer->Width/(f32)NumbersCount)));
            }
            else
            {
                Idx = ((u32)X)*(NumbersCount/Buffer->Width);
            }
            
            u32 Gray = Numbers[Idx];
            
            u32 Color = 0;
            if(SelectedSort)
            {            
                Color = ((0xFF << 3*8) |
                         ((Gray & 0xFF) << 2*8) |
                         ((Gray & 0xFF) << 1*8) |
                         ((0x0) << 0*8));
            }
            else
            {
                Color = ((0xFF << 3*8) |
                         (Gray << 2*8) |
                         ((0x0 & 0xFF) << 1*8) |
                         (0x0 << 0*8)); 
            }
            
            u32 *Pixel = PixelFromBuffer(Buffer, (u32)X, (u32)Y);
            *Pixel = Color;
        }
    }
    
    return false;
}
