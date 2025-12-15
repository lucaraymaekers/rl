#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "base/base.h"
#include "cu.h"
#include "platform.h"

PUSH_WARNINGS
#define HANDMADE_FONT_IMPLEMENTATION
#include "lib/handmade_font.h"
POP_WARNINGS

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

//~ GPU land
CU_host_shared  f32
Squared(f32 A)
{
    f32 Result = A * A;
    return Result;
}

CU_device  f32
DegreesToRadians(f32 Degrees)
{
    f32 Result = Degrees*3.14159265359f/180.0f;
    return Result;
}

CU_host_shared  f32
DistanceInsideRoundedRectangle(s32 X, s32 Y, f32 Width, f32 Height, f32 Radius)
{
    f32 Result = false;
    
    f32 hX = 0.5f*Width;
    f32 hY = 0.5f*Height;
    
    f32 dX = fabsf((f32)X - hX) - (hX - Radius);
    f32 dY = fabsf((f32)Y - hY) - (hY - Radius);
    dX = fmaxf(dX, 0.0f);
    dY = fmaxf(dY, 0.0f);
    
    Result = (Squared(Radius) - (Squared(dX) + Squared(dY)));
    
    return Result;
}

//- Random 
#define PCG_DEFAULT_MULTIPLIER_64 6364136223846793005ULL
#define PCG_DEFAULT_INCREMENT_64   1442695040888963407ULL

internal CU_host_shared void
RandomStep(random_series *Series)
{
    u64 NewState = (Series->State) * Series->Multiplier + Series->Increment;
	Series->State = NewState;
}

internal void
RandomSeed(random_series *Series, u64 Seed)
{
    Series->State = 0;
    RandomStep(Series);
    Series->State += Seed;
    RandomStep(Series);
}

internal CU_host_shared void
RandomLeap(random_series *Series, u64 Delta)
{
    u64 Increment = Series->Increment;
    u64 Multiplier = Series->Multiplier; 
    u64 AccumulatedMult = 1u;
    u64 AccumulatedInc = 0u;
    while(Delta > 0)
    {
        if(Delta & 1)
        {
            AccumulatedMult *= Multiplier;
            AccumulatedInc = AccumulatedInc * Multiplier + Increment;
        }
        Increment = (Multiplier + 1) * Increment;
        Multiplier *= Multiplier;
        Delta /= 2;
    }
    
    u64 NewState = AccumulatedMult * Series->State + AccumulatedInc;
    Series->State = NewState;
}

internal CU_host_shared u32
RandomNext(random_series *Series)
{
    u32 Result = 0;
    
    Result = (u32)((Series->State ^ (Series->State >> 22)) >> (22 + (Series->State >> 61)));
    RandomStep(Series);
    
    return Result;
}

internal CU_host_shared f32 
RandomF32(random_series *Series)
{
    f32 Result = ldexpf((f32)RandomNext(Series), -32);
    return Result;
}

internal CU_host_shared  f32
RandomUnilateral(random_series *Series)
{
    f32 Result = RandomF32(Series);
    return Result;
}

internal CU_host_shared f32
RandomBilateral(random_series *Series)
{
    f32 Result = 2.0f*RandomUnilateral(Series) - 1.0f;
    return Result;
}

internal CU_host_shared f32
RandomBetween(random_series *Series, f32 Min, f32 Max)
{
    f32 Range = Max - Min;
    f32 Result = Min + RandomUnilateral(Series)*Range;
    return Result;
}

#if 0
internal f32 
RandomDegree(random_series *Series, f32 Center, f32 Radius, f32 MaxAllowed)
{
    f32 MinVal = Center - Radius;
    if(MinVal < -MaxAllowed)
    {
        MinVal = -MaxAllowed;
    }
    
    f32 MaxVal = Center + Radius;
    if(MaxVal > MaxAllowed)
    {
        MaxVal = MaxAllowed;
    }
    
    f32 Result = RandomBetween(Series, MinVal, MaxVal);
    return Result;
}
#endif

internal CU_kernel void
GeneratePoints(random_series Series, point *Out, umm Count)
{
    s32 Idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(Idx < Count)
    {
        point Point = {0};
        
        RandomLeap(&Series, Idx);
        
        Point.Lat = RandomBetween(&Series, -90.0f, 90.0f);
        Point.Lon = RandomBetween(&Series, -180.0f, 180.0f);;
        
        Out[Idx] = Point;
    }
}

//- Rendering 
internal CU_kernel void
FillRectangle(u8 *Pixels, s32 BytesPerPixel, s32 Pitch, 
              s32 Width, s32 Height, u32 Color)
{
    s32 Idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    s32 X = Idx%Width;
    s32 Y = Idx/Width;
    
    if((X >= 0 && X < Width) &&
       (Y >= 0 && Y < Height))
    {
        u32 *Pixel = (u32 *)(Pixels + Y*Pitch + X*BytesPerPixel);
        *Pixel = Color;
    }
}

internal CU_kernel void
RenderPoints(u8 *Pixels, s32 Pitch, s32 BytesPerPixel, s32 Width, s32 Height, 
             point *Points, s32 PointsCount, app_input Input) 
{
    s32 PointRadius = 1;
    
    s32 Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < PointsCount) 
    { 
        f32 Lat = Points[Idx].Lat;
        f32 Lon = Points[Idx].Lon;
        
        // Clamp, because the mercator projection will produce infinite Y
        if(Lat > 85.0f) Lat = 85.0f;
        if(Lat < -85.0f) Lat = -85.0f;
        
        // https://en.wikipedia.org/wiki/Mercator_projection#Mathematics
        f32 Phi = DegreesToRadians(Lat);
        f32 MercY = logf(tanf(M_PI_4 + Phi/2.0f));
        
        // In pixels
        f32 X = ((Lon + 180.0f)/360.0f)*Width;
        f32 Y = ((1.0f - MercY/M_PI)/2.0f)*Height;
        s32 pX = roundf(X);
        s32 pY = roundf(Y);
        
        for(s32 dY = -PointRadius; dY <= PointRadius; dY += 1)
        {
            for(s32 dX = -PointRadius; dX <= PointRadius; dX += 1)
            {
                if(Squared(dX) + Squared(dY) <= Squared(PointRadius))
                {                    
                    s32 SX = pX + dX;
                    s32 SY = pY + dY;
                    if((SX >= 0 && SX < Width) && 
                       (SY >= 0 && SY < Height)) 
                    {
                        u32 *Pixel = (u32 *)(Pixels + SY*Pitch + SX*BytesPerPixel);
                        *Pixel = ColorPoint;
                    }
                }
            }
        }
        
    }
}

internal CU_kernel void
DrawRoundedRectangle(u8 *Pixels, s32 BytesPerPixel, s32 Pitch,
                     s32 Width, s32 Height,
                     f32 Radius,
                     u32 Color)
{
    s32 X = blockIdx.x * blockDim.x + threadIdx.x;
    s32 Y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(X < Width && Y < Height)
    {        
        f32 DistanceFromCenter = DistanceInsideRoundedRectangle(X, Y, (f32)Width, (f32)Height, Radius);
        if(DistanceFromCenter >= 0)
        {
            u32 *Pixel = (u32 *)((u8 *)Pixels + Y*Pitch + X*BytesPerPixel);
            if(DistanceFromCenter < 20)
            {
                *Pixel = 0xFFFFFFFF;
            }
            else
            {
                *Pixel = Color;
            }
        }
    }
}

//~ CPU land

internal b32
DrawButton(u8 *DevicePixels, app_offscreen_buffer *Buffer,
           app_input *Input,
           s32 X, s32 Y, s32 Width, s32 Height, f32 Radius)
{
    b32 Pressed = false;
    
    u8 *ButtonPixels = DevicePixels + Y*Buffer->Pitch + X*Buffer->BytesPerPixel;
    
    u32 Color = ColorButton;
    
    if(Input)
    {    
        s32 pX = Input->MouseX - X;
        s32 pY = Input->MouseY - Y;
        
        b32 InsideButton = (DistanceInsideRoundedRectangle(pX, pY, (f32)Width, (f32)Height, Radius) >= 0);
        b32 MousePressed = (Input->Buttons[PlatformButton_Left].EndedDown);
        
        Color = (InsideButton ? 
                 (MousePressed ?
                  ColorButtonPressed : ColorButtonHovered) :
                 ColorButton);
        
        Pressed = InsideButton && WasPressed(Input->Buttons[PlatformButton_Left]);
    }
    
    dim3 block(16, 16);
    dim3 grid((Width  + block.x - 1) / block.x,
              (Height + block.y - 1) / block.y);
    DrawRoundedRectangle<<<grid, block>>>(ButtonPixels, Buffer->BytesPerPixel, Buffer->Pitch, Width, Height, Radius, Color);
    CU_Check(cudaGetLastError());
    
    return Pressed;
}

C_LINKAGE UPDATE_AND_RENDER(UpdateAndRender)
{
    ThreadContextSelect(Context);
    
    u8 *DevicePixels = PushArray(GPUArena, u8, Buffer->Pitch*Buffer->Height);
    
    if(!App->Initialized)
    {
        point Points[] =
        {
            {0, 0},
            {85, -170},
            {-85,0},
            {0,175},
            {0,-175},
            {85,0},
        };
        u32 PointsCount = ArrayCount(Points);
        
        App->Arena = ArenaAlloc();
        
        App->Points = PushArray(App->Arena, point, PointsCount);
        for(EachIndex(Idx, PointsCount))
        {
            App->Points[Idx] = Points[Idx];
        }
        App->PointsCount = PointsCount;
        
        InitFont(&App->Font, "./data/font.ttf");
        
        App->GenerateAmount = 10;
        
        App->Series.Multiplier = PCG_DEFAULT_MULTIPLIER_64;
        App->Series.Increment  = PCG_DEFAULT_INCREMENT_64;
        RandomSeed(&App->Series, 0);
        
        App->Initialized = true;
    }
    
    point *DevicePoints = PushArray(GPUArena, point, App->PointsCount);
    CU_Check(cudaMemcpy(DevicePoints, App->Points, 
                        App->PointsCount*sizeof(point),
                        cudaMemcpyHostToDevice));
    
    // Render
    {
        // Buttons
        {
            // Generate
            b32 Pressed = 0;
            s32 Y = 10;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
            if(Pressed)
            {
                s32 GenerateAmount = App->GenerateAmount;
                s32 BlockSize = 32;
                s32 BlocksCount = GenerateAmount/BlockSize + 1;
                
                point *DevicePoints = PushArray(GPUArena, point, GenerateAmount);
                GeneratePoints<<<BlocksCount, BlockSize>>>(App->Series, DevicePoints, GenerateAmount);
                CU_Check(cudaGetLastError());
                
                point *HostPoints = PushArray(App->Arena, point, GenerateAmount);
                CU_Check(cudaMemcpy(HostPoints, DevicePoints, sizeof(point)*GenerateAmount, cudaMemcpyDeviceToHost));
                
                App->PointsCount += GenerateAmount;
                RandomLeap(&App->Series, GenerateAmount);
            }
            
            // Clear
            Y += 40;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
            if(Pressed)
            {
                App->Arena->Pos -= App->PointsCount*sizeof(point);
                App->PointsCount = 0;
            }
            
            // +10
            Y += 40;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
            if(Pressed)
            {
                App->GenerateAmount += 10;
            }
            
            // *10
            Y += 40;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
            if(Pressed)
            {
                if(App->GenerateAmount < 100000000)
                {
                    App->GenerateAmount *= 10;
                }
            }
            
            // =10
            Y += 40;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
            if(Pressed)
            {
                App->GenerateAmount = 10;
            }
            
            DrawButton(DevicePixels, Buffer, 0, 120, 10, 180, 150, 10.0f); 
        }
        
        s32 MapWidth = (s32)roundf(0.6f*(f32)Buffer->Width);
        s32 MapHeight = Buffer->Height - 20;
        
        s32 MapXOffset = Buffer->Width - MapWidth - 10;
        s32 MapYOffset = Buffer->Height - MapHeight - 10;
        u8 *MapPixels = (DevicePixels + Buffer->BytesPerPixel*(MapYOffset*Buffer->Width + MapXOffset));
        
        s32 PixelCount = MapWidth*MapHeight;
        s32 BlockSize = 32;
        s32 BlocksCount = PixelCount/BlockSize + 1;
        
        FillRectangle<<<BlocksCount, BlockSize>>>(MapPixels, Buffer->BytesPerPixel, Buffer->Pitch,  
                                                  MapWidth, MapHeight, ColorBackground);
        CU_Check(cudaGetLastError());
        
        if(App->PointsCount)
        {            
            s32 BlockSize = 32;
            s32 BlocksCount = App->PointsCount/BlockSize + 1;
            
            RenderPoints<<<BlocksCount, BlockSize>>>(MapPixels, Buffer->Pitch, Buffer->BytesPerPixel, MapWidth, MapHeight, 
                                                     DevicePoints, App->PointsCount, *Input);
            CU_Check(cudaGetLastError());
        }
        
        CU_Check(cudaDeviceSynchronize());
        CU_Check(cudaMemcpy(Buffer->Pixels, DevicePixels, Buffer->Pitch*Buffer->Height, cudaMemcpyDeviceToHost));
        
        // Show cursor
        {
            s32 pX = Input->MouseX;
            s32 pY = Input->MouseY;
            s32 PointRadius = 3;
            
            s32 MinX = MapXOffset;
            s32 MaxX = MapXOffset + MapWidth;
            s32 MinY = MapYOffset;
            s32 MaxY = MapYOffset + MapHeight;
            
            b32 InsideMap = ((pX >= MinX && pX < MaxX) &&
                             (pY >= MinY && pY < MaxY));
            if(InsideMap)
            {
                if(WasPressed(Input->Buttons[PlatformButton_Left]))
                {
                    s32 pX = Input->MouseX;
                    s32 pY = Input->MouseY;
                    
                    point *Point = App->Points + App->PointsCount;
                    App->PointsCount += 1;
                    PushStruct(App->Arena, point);
                    
                    pX -= MapXOffset;
                    pY -= MapYOffset;
                    
                    // Inverse Mercator latitude
                    f32 N = (f32)M_PI * (1.0f - 2.0f * (f32)pY / (f32)MapHeight);
                    f32 Phi = 2.0f * atanf(expf(N)) - (f32)M_PI_2;
                    
                    Point->Lon = ((f32)pX / (f32)MapWidth) * 360.0f - 180.0f;
                    Point->Lat = Phi * 180.0f / (f32)M_PI;
                }
            }
        }
        
        f32 Y = 29.0f;
        
        
        {
            f32 HeightPx = 17.0f;
            DrawText(Buffer, &App->Font, HeightPx, S8Lit("Generate"), v2{30.0f, Y}, ColorButtonText, false);
            Y += 40.0f;
            DrawText(Buffer, &App->Font, HeightPx, S8Lit("Clear"), v2{42.0f, Y}, ColorButtonText, false);
            
            Y += 40.0f;
            DrawText(Buffer, &App->Font, HeightPx, S8Lit("+10"), v2{48.0f, Y}, ColorButtonText, false);
            
            Y += 40.0f;
            DrawText(Buffer, &App->Font, HeightPx, S8Lit("*10"), v2{48.0f, Y}, ColorButtonText, false);
            
            Y += 40.0f;
            DrawText(Buffer, &App->Font, HeightPx, S8Lit("=10"), v2{48.0f, Y}, ColorButtonText, false);
            
            Y = 30.0f;
            DrawTextFormat(CPUArena, Buffer, &App->Font, 140.0f, Y, ColorBackground, "Points: %d", App->PointsCount); 
            Y += 14.0f;
            DrawTextFormat(CPUArena, Buffer, &App->Font, 140.0f, Y, ColorBackground, "Memory: %.2fKB", (f64)App->Arena->Pos/1024.0); 
            Y += 14.0f;
            DrawTextFormat(CPUArena, Buffer, &App->Font, 140.0f, Y, ColorBackground, "Generate size: %d", App->GenerateAmount); 
            
        }
        
    }
    
}

ENTRY_POINT(EntryPoint)
{
    // STUB
    return 0;
}
