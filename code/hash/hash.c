#include <lib/md5.h>

#define FORCE_THREADS_COUNT 1
#include <base/base.h>

typedef struct tile tile;
struct tile
{
    s32 X, Y, Z;
    
    b32 Init;
    tile *NextInHash;
};

typedef struct tile_map tile_map;
internal struct tile_map
{
    tile Tiles[4096];
};

internal tile *
GetTile(tile_map *Map, s32 X, s32 Y, s32 Z, arena *Arena)
{
    tile *Tile = 0;
    
    // TODO(luca): BETTER HASH FUNCTION!!!!
#if 1 
    s32 Input[3] = {X, Y, Z};
    
    local_persist md5_ctx MD5Context = {0};
    local_persist b32 Initialized = false;
    if(!Initialized)
    {
        md5_init(&MD5Context);
    }
    
    u8 Digest[MD5_DIGEST_SIZE] = {0};
    md5_update(&MD5Context, Input, sizeof(Input));
    md5_finish(&MD5Context, Digest);
    
    u32 HashValue = ((u32 *)Digest)[0];
    
#else
    u32 HashValue = 10*X + 7*Y + 3*Z;
#endif
    
    s32 HashSlot = (HashValue & (ArrayCount(Map->Tiles) - 1));
    Assert(HashSlot < ArrayCount(Map->Tiles));
    
    Tile = Map->Tiles + HashSlot;
    do 
    {
        if(Tile->Init &&
           (Tile->X == X &&
            Tile->Y == Y && 
            Tile->Z == Z))
        {
            break;
        }
        
        if(!Tile->Init && Arena)
        {
            Tile->X = X;
            Tile->Y = Y;
            Tile->Z = Z;
            Tile->Init = true;
            
            break;
        }
        
        // Allocator is set + we are at the end of the list
        if(Arena && (!Tile->NextInHash))
        {
            Tile->NextInHash = PushStruct(Arena, tile);
            Tile->Init = false;
        }
        
        Tile = Tile->NextInHash;
    } while(Tile);
    
    return Tile;
}

internal void 
GetTileAndPrint(tile_map *TileMap, s32 X, s32 Y, s32 Z, arena *Arena)
{
    tile *Tile = GetTile(TileMap, X, Y, Z, Arena);
    OS_PrintFormat("[%ld] %d,%d,%d: %p\n", LaneIndex(), Tile->X, Tile->Y, Tile->Z, Tile);
}

ENTRY_POINT(EntryPoint)
{
    arena *Arena = ArenaAlloc();
    tile_map TileMap = {0};
    
    GetTileAndPrint(&TileMap, 1, 2, 3, Arena);
    GetTileAndPrint(&TileMap, 1, 2, 4, Arena);
    GetTileAndPrint(&TileMap, 8, 2, 3, Arena);
    GetTileAndPrint(&TileMap, 8, 3, 3, Arena);
    GetTileAndPrint(&TileMap, 1, 2, 3, Arena);
    
    return 0;
}