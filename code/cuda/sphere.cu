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
#define ColorPoint 0xFF00FFFF
#define ColorCursor 0xFFFF0000
#define ColorCursorPressed ColorPoint  
#define ColorBackground 0xFF3A4151

//~ Functions

CU_device CU_host inline f32
Squared(f32 A)
{
    f32 Result = A * A;
    return Result;
}

CU_device inline f32
DegreesToRadians(f32 Degrees)
{
    f32 Result = Degrees*3.14159265359f/180.0f;
    return Result;
}

CU_kernel void
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

CU_kernel void
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

CU_device CU_host b32
IsInsideRoundedRectangle(s32 X, s32 Y, f32 Width, f32 Height, f32 Radius)
{
    b32 Result = false;
    
    f32 hX = 0.5f*Width;
    f32 hY = 0.5f*Height;
    
    f32 dX = fabsf((f32)X - hX) - (hX - Radius);
    f32 dY = fabsf((f32)Y - hY) - (hY - Radius);
    dX = fmaxf(dX, 0.0f);
    dY = fmaxf(dY, 0.0f);
    
    Result = ((Squared(dX) + Squared(dY)) <= Squared(Radius));
    
    return Result;
}

CU_kernel void
DrawRoundedRectangle(u8 *Pixels, s32 BytesPerPixel, s32 Pitch,
                     s32 Width, s32 Height,
                     f32 Radius,
                     u32 Color)
{
    s32 X = blockIdx.x * blockDim.x + threadIdx.x;
    s32 Y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(X < Width && Y < Height)
    {        
        b32 IsInside = IsInsideRoundedRectangle(X, Y, (f32)Width, (f32)Height, Radius);
        if(IsInside)
        {
            u32 *Pixel = (u32 *)((u8 *)Pixels + Y*Pitch + X*BytesPerPixel);
            *Pixel = Color;
        }
    }
}

//~ CPU land
b32
DrawButton(u8 *DevicePixels, app_offscreen_buffer *Buffer,
           app_input *Input,
           s32 X, s32 Y, s32 Width, s32 Height, f32 Radius)
{
    b32 Pressed = false;
    
    u8 *ButtonPixels = DevicePixels + Y*Buffer->Pitch + X*Buffer->BytesPerPixel;
    
    s32 pX = Input->MouseX - X;
    s32 pY = Input->MouseY - Y;
    
    b32 InsideButton = IsInsideRoundedRectangle(pX, pY, (f32)Width, (f32)Height, Radius);
    b32 MousePressed = (Input->Buttons[PlatformButton_Left].EndedDown);
    
    u32 Color = (InsideButton ? 
                 (MousePressed ?
                  ColorBackground : ColorCursorPressed) :
                 ColorCursor);
    
    dim3 block(16, 16);
    dim3 grid((Width  + block.x - 1) / block.x,
              (Height + block.y - 1) / block.y);
    DrawRoundedRectangle<<<grid, block>>>(ButtonPixels, Buffer->BytesPerPixel, Buffer->Pitch, Width, Height, Radius, Color);
    CU_Check(cudaGetLastError());
    
    Pressed = InsideButton && MousePressed;
    
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
        
        App->Points = PushArray(App->Arena, point, 1024);
        for(EachIndex(Idx, PointsCount))
        {
            App->Points[Idx] = Points[Idx];
        }
        App->PointsCount = PointsCount;
        
        InitFont(&App->Font, "./data/font.ttf");
        
        App->Initialized = true;
    }
    
    point *DevicePoints = PushArray(GPUArena, point, App->PointsCount);
    CU_Check(cudaMemcpy(DevicePoints, App->Points, 
                        App->PointsCount*sizeof(point),
                        cudaMemcpyHostToDevice));
    
    // Render
    {
#if RL_INTERNAL
#else
        // Clear
        {        
            s32 BlockSize = 32;
            s32 BlocksCount = Width*Height + 1;
            FillRectangle<<<BlocksCount, BlockSize>>>((u32 *)DevicePixels, Pitch, BytesPerPixel, Width, Height, 0);
        }
#endif
        
        {
            b32 Pressed = 0;
            s32 Y = 10;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
            
            Y += 40;
            Pressed = DrawButton(DevicePixels, Buffer, Input, 10, Y, 100, 30, 10.0f);
        }
        
        s32 MapWidth = (s32)roundf(0.6f*(f32)Buffer->Width);
        s32 MapHeight = (s32)roundf(0.8f*(f32)Buffer->Height);
        
        s32 MapXOffset = Buffer->Width - MapWidth - 10;
        s32 MapYOffset = Buffer->Height - MapHeight - 10;
        u8 *MapPixels = (DevicePixels + Buffer->BytesPerPixel*(MapYOffset*Buffer->Width + MapXOffset));
        
        s32 PixelCount = MapWidth*MapHeight;
        s32 BlockSize = 32;
        s32 BlocksCount = PixelCount/BlockSize + 1;
        
        FillRectangle<<<BlocksCount, BlockSize>>>(MapPixels, Buffer->BytesPerPixel, Buffer->Pitch,  
                                                  MapWidth, MapHeight, ColorBackground);
        CU_Check(cudaGetLastError());
        RenderPoints<<<1, App->PointsCount>>>(MapPixels, Buffer->Pitch, Buffer->BytesPerPixel, MapWidth, MapHeight, 
                                              DevicePoints, App->PointsCount, *Input);
        CU_Check(cudaGetLastError());
        
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
                for(s32 dY = -PointRadius; dY <= PointRadius; dY += 1)
                {
                    for(s32 dX = -PointRadius; dX <= PointRadius; dX += 1)
                    {
                        if(Squared((f32)dX) + Squared((f32)dY) <= Squared((f32)PointRadius))
                        {                    
                            s32 SX = pX + dX;
                            s32 SY = pY + dY;
                            if((SX >= 0 && SX < Buffer->Width) && 
                               (SY >= 0 && SY < Buffer->Height)) 
                            {
                                u32 *Pixel = (u32 *)((u8 *)Buffer->Pixels + SY*Buffer->Pitch + SX*Buffer->BytesPerPixel);
                                *Pixel = (Input->Buttons[PlatformButton_Left].EndedDown ? ColorCursorPressed : ColorCursor); 
                            }
                        }
                    }
                }
                
                if(WasPressed(Input->Buttons[PlatformButton_Left]))
                {
                    s32 pX = Input->MouseX;
                    s32 pY = Input->MouseY;
                    
                    point *Point = App->Points + App->PointsCount;
                    App->PointsCount += 1;
                    
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
        
        DrawText(Buffer, &App->Font, 16.0f, S8Lit("Generate"), v2{30.0f, 28.0f}, v3{0.0f, 0.0f, 0.0f}, false);
        DrawText(Buffer, &App->Font, 16.0f, S8Lit("+10"), v2{48.0f, 28.0f + 40.0f}, v3{0.0f, 0.0f, 0.0f}, false);
        
        
    }
    
}

ENTRY_POINT(EntryPoint)
{
    // STUB
    return 0;
}
