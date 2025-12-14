#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "base/base.h"
#include "cu.h"

CU_device inline f32
Squared(f32 A)
{
    f32 Result = A * A;
    return Result;
}

struct point
{
    f32 Lat;
    f32 Lon;
};

CU_device inline f32
DegreesToRadians(f32 Degrees)
{
    f32 Result = Degrees*3.14159265359f/180.0f;
    return Result;
}

CU_kernel void
FillRectangle(u32 *Pixels, s32 Pitch, s32 BytesPerPixel, s32 Width, s32 Height,
              u32 Color)
{
    s32 Idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    s32 X = Idx%Width;
    s32 Y = Idx/Width;
    
    if((X >= 0 && X < Width) &&
       (Y >= 0 && Y < Height))
    {
        u32 *Pixel = (u32 *)((u8 *)Pixels + Y*Pitch + X*BytesPerPixel);
        *Pixel = Color;
    }
}

CU_kernel void
RenderPoints(u32 *Pixels, s32 Pitch, s32 BytesPerPixel, s32 Width, s32 Height, 
             point *Points, s32 PointsCount) 
{
    s32 PointRadius = 3;
    
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
        f32 MercN = logf(tanf(M_PI_4 + Phi/2.0f));
        
        // In pixels
        f32 X = ((Lon + 180.0f)/360.0f)*Width;
        f32 Y = ((1.0f - MercN/M_PI)/2.0f)*Height;
        s32 PX = roundf(X);
        s32 PY = roundf(Y);
        
        for(s32 dY = -PointRadius; dY <= PointRadius; dY += 1)
        {
            for(s32 dX = -PointRadius; dX <= PointRadius; dX += 1)
            {
                if(Squared(dX) + Squared(dY) <= Squared(PointRadius))
                {                    
                    s32 SX = PX + dX;
                    s32 SY = PY + dY;
                    if((SX >= 0 && SX < Width) && 
                       (SY >= 0 && SY < Height)) 
                    {
                        u32 *Pixel = (u32 *)((u8 *)Pixels + SY*Pitch + SX*BytesPerPixel);
                        *Pixel = 0xFF0000FF; // red 
                    }
                }
            }
        }
    }
}

C_LINKAGE CU_UPDATE_AND_RENDER(UpdateAndRender)
{
    ThreadContextSelect(Context);
    
    s32 Size = Width*Height*BytesPerPixel;
    u8 *DevicePixels = PushArray(GPUArena, u8, Size);
    
    point Points[] =
    {
        {0, 0},
        {80, -170},
        {-80,0},
        {0,170},
        {0,-170},
        {80,0},
    };
    s32 NumPoints = (s32)ArrayCount(Points);
    
    point *DevicePoints = PushArray(GPUArena, point, NumPoints);
    CU_Check(cudaMemcpy(DevicePoints, Points, sizeof(Points), cudaMemcpyHostToDevice));
    
    // Render
    {    
        s32 MapWidth = (s32)roundf(0.6f*(f32)Width);
        s32 MapHeight = (s32)roundf(0.8f*(f32)Height);
        
        s32 MapXOffset = Width - MapWidth - 10;
        s32 MapYOffset = Height - MapHeight - 10;
        u32 *MapPixels = (u32 *)(DevicePixels + BytesPerPixel*(MapYOffset*Width + MapXOffset));
        
        s32 PixelCount = MapWidth*MapHeight;
        s32 BlockSize = 32;
        s32 BlocksCount = PixelCount/BlockSize + 1;
        
        FillRectangle<<<BlocksCount, BlockSize>>>(MapPixels, Pitch, BytesPerPixel, MapWidth, MapHeight, 0xFF51413A);
        CU_Check(cudaGetLastError());
        RenderPoints<<<1, NumPoints>>>(MapPixels, Pitch, BytesPerPixel, MapWidth, MapHeight, 
                                       DevicePoints, NumPoints);
        CU_Check(cudaGetLastError());
        
        CU_Check(cudaDeviceSynchronize());
        CU_Check(cudaMemcpy(HostPixels, DevicePixels, Size, cudaMemcpyDeviceToHost));
    }
    
}

ENTRY_POINT(EntryPoint)
{
    // STUB
    return 0;
}
