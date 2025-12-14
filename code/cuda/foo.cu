#define FORCE_THREADS_COUNT 1
#include "base/base.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
PUSH_WARNINGS;
#include "lib/stb_image_write.h"
POP_WARNINGS;

#include "cu.h"

CU_kernel void 
FillRectangle(u32 *Buffer, s32 Width, s32 Height, s32 Pitch, u32 Color)
{
    s32 Idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    s32 X = Idx % Width;
    f32 Fact = 255.0f/(f32)Width;
    s32 RedValue = roundf(Fact*X);
    
    // NOTE(luca): AA BB GG RR
    Color = ((0xFF     << 3*8) |
             (0x00     << 2*8) |
             (RedValue << 1*8) |
             (0xFF     << 0*8));
    
    Buffer[Idx] = Color;
}

ENTRY_POINT(EntryPoint)
{
    arena *Scratch = GetScratch();
    
    s32 BlockSize = 32;
    
    s32 Width = 1920;
    s32 Height = 1080;
    s32 BytesPerPixel = 4;
    s32 Pitch = Width*BytesPerPixel;
    
    umm Size = PadSize(Pitch*Height, BlockSize*BytesPerPixel);
    u32 *Buffer = (u32 *)ArenaPushAligned(Scratch, Size, sizeof(u32));
    
    // CUDA init
    CU_Check(cudaSetDevice(0));
    cudaDeviceProp Prop;
    CU_Check(cudaGetDeviceProperties(&Prop, 0));
    arena *GPUArena = CU_ArenaAlloc(Scratch);
    u32 *DeviceBuffer = (u32 *)ArenaPushAligned(GPUArena, Size, sizeof(u32));
    
    s32 BlocksCount = (s32)(Size/BlockSize/BytesPerPixel);
    FillRectangle<<<BlocksCount, BlockSize>>>(DeviceBuffer, Width, Height, BytesPerPixel, 0xFF00FF00);
    CU_Check(cudaGetLastError());
    CU_Check(cudaDeviceSynchronize());
    CU_Check(cudaMemcpy(Buffer, DeviceBuffer, Size, cudaMemcpyDeviceToHost));
    
    
    
    stbi_write_bmp("out.bmp", Width, Height, 4, Buffer);
    
    return 0;
}