#define RL_BASE_NO_ENTRYPOINT 1
#include "base/base.h"
#include "base/base.c"

#include "ex_random.h"
#include "ex_math.h"
#include "ex_platform.h"
#include "ex_libs.h"
#include "ex_gl.h"

#define U32ToV3Arg(Hex) \
((f32)((Hex >> 8*2) & 0xFF)/255.0f), \
((f32)((Hex >> 8*1) & 0xFF)/255.0f), \
((f32)((Hex >> 8*0) & 0xFF)/255.0f)

#define ColorList \
SET(Text,             0xff87bfcf) \
SET(Point,            0xff00ffff) \
SET(Cursor,           0xffff0000) \
SET(Button,           0xff0172ad) \
SET(ButtonHovered,    0xff017fc0) \
SET(ButtonPressed,    0xff0987c8) \
SET(ButtonText,       0xfffbfdfe) \
SET(Background,       0xff13171f) \
SET(BackgroundSecond, 0xff3a4151) \
SET(Red,              0xffbf616a) \
SET(Green,            0xffa3be8c) \
SET(Orange,           0xffd08770) \
SET(Magenta,          0xffb48ead) \
SET(Yellow,           0xffebcb8b)

#define SET(Name, Value) u32 ColorU32_##Name = Value; 
ColorList
#undef SET

#define SET(Name, Value) v3 Color_##Name = {U32ToV3Arg(Value)};
ColorList
#undef SET

#define Path_Code                 ".." SLASH "code" SLASH "example" SLASH 

struct app_state
{
    
};

internal app_offscreen_buffer 
LoadImage(arena *Arena, str8 ExeDirPath, str8 Path)
{
    app_offscreen_buffer Result = {};
    
    char *FilePath = PathFromExe(Arena, ExeDirPath, Path);
    str8 File = OS_ReadEntireFileIntoMemory(FilePath);
    s32 Width, Height, Components;
    s32 BytesPerPixel = 4;
    u8 *Image = stbi_load_from_memory(File.Data, (int)File.Size, &Width, &Height, &Components, BytesPerPixel);
    Assert(Components == BytesPerPixel);
    
    Result.Width = Width;
    Result.Height = Height;
    Result.BytesPerPixel = BytesPerPixel;
    Result.Pitch = Result.BytesPerPixel*Width;
    Result.Pixels = Image;
    
    return Result;
}

#define AssetPath(FileName) S8("../data/game/" FileName)

C_LINKAGE
UPDATE_AND_RENDER(UpdateAndRender)
{
    b32 ShouldQuit = false;
    
#if RL_INTERNAL    
    GlobalDebuggerIsAttached = Memory->IsDebuggerAttached;
#endif
    
    local_persist s32 GLADVersion = gladLoaderLoadGL();
    
    if(!Memory->Initialized)
    {
        Memory->AppState = PushStruct(PermanentArena, app_state);
        
        Memory->Initialized = true;
    }
    
    for EachIndex(Idx, Input->Text.Count)
    {
        app_text_button Key = Input->Text.Buffer[Idx];
        
        if(!Key.IsSymbol)
        {
            if(Key.Codepoint == 'b')
            {
                DebugBreak;
            }
        }
        else
        {
            if(Key.Codepoint == PlatformKey_Escape)
            {
                ShouldQuit = true;
            }
        }
    }
    
    local_persist app_offscreen_buffer EnemyImage = LoadImage(PermanentArena, Memory->ExeDirPath, AssetPath("enemy.png"));
    
    gl_handle VAOs[1] = {};
    gl_handle VBOs[2] = {};
    gl_handle Textures[1] = {};
    glGenVertexArrays(ArrayCount(VAOs), &VAOs[0]);
    glGenBuffers(ArrayCount(VBOs), &VBOs[0]);
    glGenTextures(ArrayCount(Textures), &Textures[0]);
    glBindVertexArray(VAOs[0]);
    
    glViewport(0, 0, Buffer->Width, Buffer->Height);
    glClearColor(V3Arg(Color_BackgroundSecond), 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    v2 BaseEnemyPos = V2(-0.90f, 1.0f);
    v2 EnemyDim = V2(0.1f, 0.1f);
    
    gl_handle Shader;
    
    // Render enemy
    s32 EnemiesCount = 304;
    s32 VerticesCount = EnemiesCount*6;
    v3 *Vertices = PushArray(FrameArena, v3, VerticesCount);
    v2 *TexCoords = PushArray(FrameArena, v2, VerticesCount);
    Shader = gl_ProgramFromShaders(FrameArena, Memory->ExeDirPath, 
                                   S8(Path_Code "vert_text.glsl"),
                                   S8(Path_Code "frag_text.glsl"));
    glUseProgram(Shader);
    
    v3 *VerticesPtr = Vertices;
    v2 *TexCoordsPtr = TexCoords;
    
    for EachIndex(Idx, EnemiesCount)
    {
        s32 EnemiesCountPerRow = 16;
        v2 EnemyPos;
        EnemyPos.X = BaseEnemyPos.X + (f32)(Idx % EnemiesCountPerRow) * (EnemyDim.X + 0.01f);
        EnemyPos.Y = BaseEnemyPos.Y - ((f32)(Idx / EnemiesCountPerRow) * EnemyDim.Y);
        
        
        gl_LoadTextureFromImage(Textures[0], EnemyImage.Width, EnemyImage.Height, EnemyImage.Pixels,
                                GL_RGBA, Shader);
        MakeQuadV2(TexCoordsPtr, V2(0.0f, 1.0f), V2(1.0f, 0.0f));
        
        v2 Min = V2(EnemyPos.X, EnemyPos.Y - EnemyDim.Y);
        v2 Max = V2AddV2(Min, EnemyDim);
        MakeQuadV3(VerticesPtr, Min, Max, -1.0f);
        
        VerticesPtr += 6;
        TexCoordsPtr += 6;
    }
    
#if 1 
    MemorySet(EnemyImage.Pixels, 0, EnemyImage.Pitch*EnemyImage.Height);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
#else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
    glEnable(GL_DEPTH_TEST);
#endif
    
    gl_LoadFloatsIntoBuffer(VBOs[0], Shader, "pos", VerticesCount, 3, Vertices);
    gl_LoadFloatsIntoBuffer(VBOs[1], Shader, "tex", VerticesCount, 2, TexCoords);
    
    glDrawArrays(GL_TRIANGLES, 0, VerticesCount);
    
    // Cleanup
    {    
        glDeleteTextures(ArrayCount(Textures), &Textures[0]);
        glDeleteBuffers(ArrayCount(VBOs), &VBOs[0]);
        glDeleteProgram(Shader);
    }
    
    return ShouldQuit;
}