#include "base/base.h"
#include "platform.h"

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
    
    if(!App->Initialized)
    {
        App->Initialized = true;
    }
    
    for(EachIndex(Y, Buffer->Height))
    {
        for(EachIndex(X, Buffer->Width))
        {
            u32 *Pixel = (u32 *)(Buffer->Pixels + Y*Buffer->Pitch + X*Buffer->BytesPerPixel);
            *Pixel = ColorMapBackground;
        }
        
    }
    
}