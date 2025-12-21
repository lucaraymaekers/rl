internal arena *
ArenaAlloc_(arena_alloc_params Params)
{
    arena *Arena = 0;
    
    umm Size = Params.DefaultSize;
    if(Params.Size)
    {
        Size = Params.Size;
    }
    
    void *Base = OS_Allocate(Size);
    
    Arena = (arena *)Base;
    Arena->Base = Base;
    Arena->Pos = sizeof(arena);
    Arena->Size = Size;
    
    return Arena;
}

internal void *
ArenaPush(arena *Arena, umm Size)
{
    void *Result = (u8 *)Arena->Base + Arena->Pos;
    
    Assert(Arena->Pos + Size < Arena->Size);
    Arena->Pos += Size;
    
    return Result;
}

internal void *
ArenaPushAligned(arena *Arena, umm Size, umm Alignment)
{
    void *Result = 0;
    
    u8 *Base = (u8 *)Arena->Base + Arena->Pos;
    umm BaseAddress = (umm)Base;
    
    umm Leftover = (BaseAddress % Alignment);
    Arena->Pos += (Alignment - Leftover);
    
    Result = ArenaPush(Arena, Size);
    
    return Result;
}

internal umm 
PadSize(umm Size, umm Padding)
{
    umm Result = Size;
    
    umm Leftover = Size%Padding;
    Result += (Padding - Leftover);
    Assert(Result % Padding == 0);
    
    return Result;
}