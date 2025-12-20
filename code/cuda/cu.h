/* date = December 12th 2025 2:58 pm */

#ifndef CU_H
#define CU_H

#include <cuda_runtime.h>
#include "platform.h"

#define CU_device __device__
#define CU_host_shared __device__ __host__ 
#define CU_kernel __global__
#define CU_static_shared __shared__
#define CU_dynamic_shared extern __shared__
#define CU_cluster __cluster_dims__

#if RL_INTERNAL
# define GPU_Assert(Expression) if(!(Expression)) { *(int *)0 = 0; }
#else
# define GPU_Assert(Expression)
#endif

#if RL_INTERNAL
# define CU_Check(Expression) { CU_Check_((Expression), __FILE__, __LINE__); }
#else
# define CU_Check(Expression) Expression
#endif

#define CU_ArenaAllocDefaultSize Megabytes(32)

//- API 
#define CU_ArenaAlloc(Arena, ...) CU_ArenaAlloc_(Arena, (arena_alloc_params){.DefaultSize = CU_ArenaAllocDefaultSize, ##__VA_ARGS__})
internal arena *CU_ArenaAlloc_(arena *CPUArena, arena_alloc_params Params);
internal inline void CU_Check_(cudaError_t Code, char *FileName, s32 Line);

//- Checked functions 
#define CU_MemoryCopy(Dest, Source, Size, Type) CU_Check(cudaMemcpy(Dest, Source, Size, Type))
#define CU_GetLastError()                       CU_Check(cudaGetLastError())
#define CU_DeviceSynchronize()                  CU_Check(cudaDeviceSynchronize())
#define CU_SetDevice(Device)                    CU_Check(cudaSetDevice(Device))
#define CU_GetDeviceProperties(Prop, Device)    CU_Check(cudaGetDeviceProperties(Prop, Device))
#define CU_Malloc(Base, Size)                   CU_Check(cudaMalloc(Base, Size))

//- Implementation 

internal inline void 
CU_Check_(cudaError_t Code, char *FileName, s32 Line)
{
    if(Code != cudaSuccess) 
    {
        OS_PrintFormat("%s(%d): ERROR: %s\n", FileName, Line, cudaGetErrorString(Code));
        DebugBreak;
    }
}

internal arena *
CU_ArenaAlloc_(arena *CPUArena, arena_alloc_params Params)
{
    umm Size = Params.DefaultSize;
    if(Params.Size)
    {
        Size = Params.Size;
    }
    
    arena *Arena = PushStruct(CPUArena, arena);
    
    void *Base = 0;
    CU_Malloc(&Base, Size);
    
    Arena->Base = Base;
    Arena->Pos = 0;
    Arena->Size = Size;
    
    return Arena;
}

#endif //CU_H