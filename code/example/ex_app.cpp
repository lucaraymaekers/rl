#define BASE_NO_ENTRYPOINT
#include "base/base.h"
#include "ex_platform.h"

#include "rl_libs.h"

NO_WARNINGS_BEGIN
#include "ex_font.h"
NO_WARNINGS_END

typedef unsigned int gl_handle;

typedef s32 face[4][3];

#define New(type, Name, Array, Count) type *Name = Array + Count; Count += 1;

//~ Constants

// AA BB GG RR
#define Color_Text          0xff87bfcf
#define Color_Point         0xFF00FFFF
#define Color_Cursor        0xFFFF0000
#define Color_Button        0xFF0172AD
#define Color_ButtonHovered 0xFF017FC0
#define Color_ButtonPressed 0xFF0987C8
#define Color_ButtonText    0xFFFBFDFE
#define Color_Background    0xFF13171F
#define Color_BackgroundSecond 0xFF3A4151

#define Strs_CodePath           ".." SLASH "code" SLASH "example" 
#define Strs_DataPath           ".." SLASH "data" 
#define Strs_FragmentShaderPath Strs_CodePath SLASH "frag.glsl"
#define Strs_VertexShaderPath   Strs_CodePath SLASH "vert.glsl"
#define Strs_TextVertexShaderPath     Strs_CodePath SLASH "vert_text.glsl"
#define Strs_TextFragmentShaderPath   Strs_CodePath SLASH "frag_text.glsl"
#define Strs_ButtonVertexShaderPath     Strs_CodePath SLASH "vert_button.glsl"
#define Strs_ButtonFragmentShaderPath   Strs_CodePath SLASH "frag_button.glsl"
#define Strs_ModelPath          Strs_DataPath SLASH "bonhomme.obj"
#define Strs_TexturePath        Strs_DataPath SLASH "bonhomme.png"

#define HexToRGBV3(Hex) \
((f32)((Hex >> 8*2) & 0xFF)/255.0f), \
((f32)((Hex >> 8*1) & 0xFF)/255.0f), \
((f32)((Hex >> 8*0) & 0xFF)/255.0f)


//~ Helpers
internal inline v2 
V2AddV2(v2 A, v2 B)
{
    v2 Result = {};
    Result.X = A.X + B.X;
    Result.Y = A.Y + B.Y;
    return Result;
}

internal inline v2 
V2AddF32(v2 A, f32 B)
{
    v2 Result = {};
    Result.X = A.X + B;
    Result.Y = A.Y + B;
    return Result;
}

internal inline v2 
V2MulF32(v2 A, f32 B)
{
    v2 Result = {};
    Result.X = A.X * B;
    Result.Y = A.Y * B;
    return Result;
}

internal inline v2
V2SubV2(v2 A, v2 B)
{
    v2 Result = {};
    Result.X = A.X - B.X;
    Result.Y = A.Y - B.Y;
    return Result;
}

internal inline v2
V2S32(s32 X, s32 Y)
{
    v2 Result = {};
    Result.X = (f32)X;
    Result.Y = (f32)Y;
    return Result;
}

internal inline v2
V2MulV2(v2 A, v2 B)
{
    v2 Result = {};
    Result.X = A.X * B.X;
    Result.Y = A.Y * B.Y;
    return Result;
}

internal inline b32
InBounds(v2 A, v2 Min, v2 Max)
{
    b32 Result = !!((A.X >= Min.X && A.X < Max.X) &&
                    (A.Y >= Min.Y && A.Y < Max.Y));
    return Result;
}

//- 

internal inline u32 *
PixelFromBuffer(app_offscreen_buffer *Buffer, s32 X, s32 Y)
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
GLErrorStatus(gl_handle Handle, b32 IsShader)
{
    b32 Success = true;
    
    char InfoLog[KB(2)] = {};
    if(IsShader)
    {
        glGetShaderiv(Handle, GL_COMPILE_STATUS, &Success);
        glGetShaderInfoLog(Handle, sizeof(InfoLog), NULL, InfoLog);
    }
    else
    {
        glGetProgramiv(Handle, GL_LINK_STATUS, &Success);
        glGetProgramInfoLog(Handle, sizeof(InfoLog), NULL, InfoLog);
    }
    
    if(!Success)
    {
        ErrorLog("%s", InfoLog);
        DebugBreak;
    }
}

internal gl_handle
CompileShaderFromSource(arena *Arena, app_state *App, str8 FileNameAfterExe, s32 Type)
{
    gl_handle Handle = glCreateShader(Type);
    
    char *FileName = PathFromExe(Arena, App, FileNameAfterExe);
    str8 Source = OS_ReadEntireFileIntoMemory(FileName);
    
    if(Source.Size)
    {    
        glShaderSource(Handle, 1, (char **)&Source.Data, NULL);
        glCompileShader(Handle);
        GLErrorStatus(Handle, true);
    }
    
    OS_FreeFileMemory(Source);
    
    return Handle;
}

internal gl_handle
ProgramFromShaders(arena *Arena, app_state *App, str8 VertPath, str8 FragPath)
{
    gl_handle Program = 0;
    
    gl_handle VertexShader, FragmentShader;
    VertexShader = CompileShaderFromSource(Arena, App, VertPath, GL_VERTEX_SHADER);
    FragmentShader = CompileShaderFromSource(Arena, App, FragPath, GL_FRAGMENT_SHADER);
    
    Program = glCreateProgram();
    glAttachShader(Program, VertexShader);
    glAttachShader(Program, FragmentShader);
    glLinkProgram(Program);
    GLErrorStatus(Program, false);
    
    glDeleteShader(FragmentShader); 
    glDeleteShader(VertexShader);
    
    return Program;
}

internal void
LoadTextureFromImage(gl_handle Texture, s32 Width, s32 Height, u8 *Image, s32 Format, gl_handle ShaderProgram)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Texture);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Width, Height, 0, Format, GL_UNSIGNED_BYTE, Image);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    // TODO(luca): Use mipmap
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    f32 Color[] = { 1.0f, 0.0f, 0.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, Color);
    
    gl_handle UTexture = glGetUniformLocation(ShaderProgram, "Texture"); 
    glUniform1i(UTexture, 0);
}

//~ Sorting
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
    
    if(Toggle)
    {
        MemoryCopy(Temp, In, sizeof(u32)*Count);
        Swap(In, Temp);
    }
    
}

//~ Parsing

internal inline 
b32 IsDigit(u8 Char)
{
    b32 Result = (Char >= '0' && Char <= '9');
    return Result;
}

internal str8
ParseName(str8 In, umm *At)
{
    str8 Result = {};
    
    umm Start = *At;
    while(!(In.Data[*At] == ' ' || In.Data[*At] == '\n')) *At += 1;
    Result = S8FromTo(In, Start, *At);
    
    return Result;
}

internal f32
ParseFloat(str8 Inner, u64 *Cursor)
{
    str8 In = S8From(Inner, *Cursor);
    
    b32 Negative = false;
    f32 Value = 0;
    
    u64 At = 0;
    
    if(In.Data[At] == '-')
    {
        At += 1;
        Negative = true;
    }
    
    // Integer part
    while(At < In.Size && IsDigit(In.Data[At])) 
    {
        f32 Digit = (f32)(In.Data[At] - '0');
        Value = 10.0f*Value + Digit;
        At += 1;
    }
    
    if(In.Data[At] == '.')
    {
        At += 1;
        
        // Fractional part
        f32 Divider = 10;
        while(At < In.Size && IsDigit(In.Data[At]))
        {
            f32 Digit = (f32)(In.Data[At] - '0');
            
            Value += (Digit / Divider);
            Divider *= 10;
            
            At += 1;
        }
    }
    
    if(Negative)
    {
        Value = -Value;
    }
    
    while(At < In.Size && In.Data[At] == ' ') At += 1; 
    
    *Cursor += At;
    
    return Value;
}

internal void 
ResetApp(app_state *App)
{
    App->Angle.X = 0.0f;
    App->Angle.Y = 0.0f;
    App->Offset.X =  0.0f;
    App->Offset.Y =  0.0f;
    App->Offset.Z = 3.0f;
}

C_LINKAGE 
UPDATE_AND_RENDER(UpdateAndRender)
{
    b32 ShouldQuit = false;
    
    OS_ProfileInit();
    
    ThreadContextSelect(Context);
    
#if RL_INTERNAL    
    GlobalDebuggerIsAttached = App->DebuggerAttached;
#endif
    
    local_persist s32 GLADVersion = 0;
    // Init
    {    
        if(App->Initialized && App->Reloaded)
        {
            GLADVersion = gladLoaderLoadGL();
            App->Reloaded = false;
        }
        
        if(!App->Initialized)
        {
            RandomSeed(&App->Series, 0);
            GLADVersion = gladLoaderLoadGL();
            
            App->Offset.Z = 3.0f;
            
            App->Initialized = true;
        }
        
#if !RL_INTERNAL    
        GLADDisableCallbacks();
#endif
    }
    OS_ProfileAndPrint("Init");
    
    local_persist b32 Animate = false;
    
    //Input 
    for EachIndex(Idx, Input->Text.Count)
    {
        app_text_button Key = Input->Text.Buffer[Idx];
        
        if(!Key.IsSymbol)
        {
            if(Key.Codepoint == 'b')
            {
                DebugBreak;
            }
            if(Key.Codepoint == 'r') 
            {
                ResetApp(App);
            }
            if(Key.Codepoint == 'a') Animate = !Animate;
            if(Key.Codepoint  == ' ')
            {
                if(Key.Modifiers & PlatformKeyModifier_Shift) App->Offset.Y += 0.1f; 
                else                                          App->Offset.Y -= 0.1f; 
            }
        }
        else
        {
            if(!(Key.Modifiers & PlatformKeyModifier_Shift))
            {
                if(Key.Codepoint == PlatformKey_Down) App->Offset.Z += 0.1f; 
                if(Key.Codepoint == PlatformKey_Up) App->Offset.Z -= 0.1f; 
                if(Key.Codepoint == PlatformKey_Left) App->Offset.X += 0.1f; 
                if(Key.Codepoint == PlatformKey_Right) App->Offset.X -= 0.1f; 
            }
            else
            {                
                if(Key.Codepoint == PlatformKey_Up)     App->Angle.Y -= 0.1f;
                if(Key.Codepoint == PlatformKey_Down)   App->Angle.Y += 0.1f;
                if(Key.Codepoint == PlatformKey_Right)  App->Angle.X += 0.1f;
                if(Key.Codepoint == PlatformKey_Left)   App->Angle.X -= 0.1f;
            }
            
            if(Key.Codepoint == PlatformKey_Escape) ShouldQuit = true;
        }
    }
    
    if(Animate)
    {
        App->Angle.X += Input->dtForFrame*4.0f;
    }
    
    App->Angle.X -= 2.0f*!!(App->Angle.X > 2.0f);
    App->Angle.X += 2.0f*!!(App->Angle.X < -2.0f);
    App->Angle.Y -= 2.0f*!!(App->Angle.Y > 2.0f);
    App->Angle.Y += 2.0f*!!(App->Angle.Y < -2.0f);
    
    umm Count = 0;
    v3 *Vertices = 0;
    v2 *TexCoords = 0; 
    v3 *Normals = 0;
    
    // Read obj file
    {    
        char *FileName = PathFromExe(FrameArena, App, S8(Strs_ModelPath));
        str8 In = OS_ReadEntireFileIntoMemory(FileName);
        
        s32 InVerticesCount = 0;
        s32 InTexCoordsCount = 0;
        s32 InNormalsCount = 0;
        s32 InFacesCount = 0;
        v3 *InVertices = PushArray(FrameArena, v3, In.Size);
        v2 *InTexCoords = PushArray(FrameArena, v2, In.Size);
        v3 *InNormals = PushArray(FrameArena, v3, In.Size);
        face *InFaces = PushArray(FrameArena, face, In.Size);
        
        if(In.Size)
        {
            for EachIndex(At, In.Size)
            {
                umm Start = At;
                
                if(In.Data[At] != '#')
                {
                    while(At < In.Size && !(In.Data[At] == ' ' || In.Data[At] == '\n')) At += 1;
                }
                
                if(In.Data[At] == ' ')
                {                    
                    str8 Keyword = S8FromTo(In, Start, At);
                    At += 1; // skip space
                    
                    if(0) {}
                    else if(S8Match(Keyword, S8("o"), false))
                    {
                        str8 Name = ParseName(In, &At);
                    }
                    else if(S8Match(Keyword, S8("v"), false))
                    {
                        New(v3, Vertex, InVertices, InVerticesCount);
                        Vertex->X = ParseFloat(In, &At);
                        Vertex->Y = ParseFloat(In, &At);
                        Vertex->Z = ParseFloat(In, &At);
                    }
                    else if(S8Match(Keyword, S8("vt"), false))
                    {
                        New(v2, TexCoord, InTexCoords, InTexCoordsCount);
                        TexCoord->X = ParseFloat(In, &At);
                        // NOTE(luca): When looking at the texture in renderdoc I noticed it was flipped.
                        TexCoord->Y = 1.0f - ParseFloat(In, &At);
                    }
                    else if(S8Match(Keyword, S8("vn"), false))
                    {
                        New(v3, Normal, InNormals, InNormalsCount);
                        Normal->X = ParseFloat(In, &At);
                        Normal->Y = ParseFloat(In, &At);
                        Normal->Z = ParseFloat(In, &At);
                    }
                    else if(S8Match(Keyword, S8("f"), false))
                    {
                        New(face, InFace, InFaces, InFacesCount);
                        for EachIndex(Y, 4)
                        {
                            for EachIndex(X, 3)
                            {
                                s32 Value = 0;
                                // Integer part
                                while(At < In.Size && IsDigit(In.Data[At])) 
                                {
                                    s32 Digit = (s32)(In.Data[At] - '0');
                                    Value = 10*Value + Digit;
                                    At += 1;
                                }
                                // NOTE(luca): Index start at 1 in obj file
                                (*InFace)[Y][X] = Value - 1;
                                
                                if(In.Data[At] == '/' || In.Data[At] == ' ') At += 1;
                            }
                            if(In.Data[At] == '\n') break;
                        }
                        
                    }
                    else if(S8Match(Keyword, S8("mtllib"), false))
                    {
                        str8 Name = ParseName(In, &At);
                    }
                    else if(S8Match(Keyword, S8("usemtl"), false))
                    {
                        str8 Name = ParseName(In, &At);
                    }
                }
                
                while(At < In.Size && In.Data[At] != '\n') At += 1;
            }
        }
        
        // Convert faces to triangles.
        {        
            // TODO(luca): This should depend on the number of vertices per face.  Blockbench always produces 4 so for now, we create a quad out of two triangles
            s32 Indices[] = {0, 1, 2, 0, 2, 3};
            s32 IndicesCount = ArrayCount(Indices);
            
            Count = InFacesCount*IndicesCount;
            Vertices = PushArray(FrameArena, v3, Count);
            TexCoords = PushArray(FrameArena, v2, Count);
            Normals = PushArray(FrameArena, v3, Count);
            
            for EachIndex(FaceIdx, InFacesCount)
            {
                for EachIndex(Idx, IndicesCount)
                {
                    s32 Index = Indices[Idx];
                    
                    s32 *FaceIndices = InFaces[FaceIdx][Index];
                    s32 vIdx = FaceIndices[0];
                    s32 vtIdx = FaceIndices[1];
                    s32 vnIdx = FaceIndices[2];
                    
                    Assert(vIdx >= 0 && vIdx < InVerticesCount);
                    Assert(vtIdx >= 0 && vtIdx < InTexCoordsCount);
                    Assert(vnIdx >= 0 && vnIdx < InNormalsCount);
                    
                    umm Offset = IndicesCount*FaceIdx + Idx;
                    Vertices[Offset] = InVertices[vIdx];
                    TexCoords[Offset] = InTexCoords[vtIdx];
                    Normals[Offset] = InNormals[vnIdx];
                }
            }
        }
        
        OS_FreeFileMemory(In);
        OS_ProfileAndPrint("Obj read");
    }
    
    s32 Major, Minor;
    gl_handle VAO, VBO[2], Tex[2], ShaderProgram, TextShader, ButtonShader;
    gl_handle PosAttrib, TexAttrib, UOffset, UAngle, UColor;
    // Setup
    { 
        glGetIntegerv(GL_MAJOR_VERSION, &Major);
        glGetIntegerv(GL_MINOR_VERSION, &Minor);
        
        glViewport(0, 0, Buffer->Width, Buffer->Height);
        
        ShaderProgram = ProgramFromShaders(FrameArena, App, 
                                           S8(Strs_VertexShaderPath),
                                           S8(Strs_FragmentShaderPath));
        glUseProgram(ShaderProgram);
    }
    
    glGenVertexArrays(1, &VAO); 
    glBindVertexArray(VAO);
    
    glGenTextures(2, &Tex[0]);
    
    glGenBuffers(2, &VBO[0]);  
    {        
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v3)*Count, Vertices, GL_STATIC_DRAW);
        
        PosAttrib = glGetAttribLocation(ShaderProgram, "pos");
        glEnableVertexAttribArray(PosAttrib);
        glVertexAttribPointer(PosAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(v3), 0);
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v2)*Count, TexCoords, GL_STATIC_DRAW);
        
        TexAttrib = glGetAttribLocation(ShaderProgram, "tex");
        glEnableVertexAttribArray(TexAttrib);
        glVertexAttribPointer(TexAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(v2), 0);
    }
    
    UOffset = glGetUniformLocation(ShaderProgram, "offset");
    UAngle = glGetUniformLocation(ShaderProgram, "angle");
    UColor = glGetUniformLocation(ShaderProgram, "color");
    
    b32 Fill = true;
    s32 Mode = (Fill) ? GL_FILL : GL_LINE;
    glPolygonMode(GL_FRONT_AND_BACK, Mode);
    
    // Draw model
    {    
        glEnable(GL_DEPTH_TEST);
        
        v3 Color = {HexToRGBV3(Color_Point)};
        f32 XAngle = Pi32 * App->Angle.X;
        f32 YAngle = Pi32 * App->Angle.Y;
        glUniform2f(UAngle, XAngle, YAngle);
        glUniform3f(UOffset, App->Offset.X, App->Offset.Y, App->Offset.Z);
        glUniform3f(UColor, Color.X, Color.Y, Color.Z);
        
        // TODO(luca): Proper assets/texture loading.
        {
            local_persist int Width, Height, Components;
            local_persist u8 *Image = 0;
            char *ImagePath = PathFromExe(FrameArena, App, S8(Strs_TexturePath));
            if(!Image) 
            {
                Image = stbi_load(ImagePath, &Width, &Height, &Components, 0);
            }
            
            LoadTextureFromImage(Tex[0], Width, Height, Image, GL_RGBA, ShaderProgram);
        }
        glClearColor(HexToRGBV3(Color_BackgroundSecond), 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPointSize(10.0f);
        glLineWidth(1.0f);
        
        glDrawArrays(GL_TRIANGLES, 0, (int)Count);
    }
    
    // NOTE(luca): For software rasterization
    app_offscreen_buffer TextImage = {};
    TextImage.Width = 960;
    TextImage.Height = 960;
    TextImage.BytesPerPixel = 4;
    TextImage.Pitch = TextImage.BytesPerPixel*TextImage.Width;
    umm Size = TextImage.BytesPerPixel*(TextImage.Height*TextImage.Width);
    TextImage.Pixels = PushArray(FrameArena, u8, Size);
    MemorySet(TextImage.Pixels, 0, Size);
    
    // Draw Button
    {
        glEnable(GL_BLEND);  
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
        glDisable(GL_DEPTH_TEST);
        
        v2 Min = {0.45f, 0.31f};
        v2 Max = {Min.X + 0.11f, Min.Y + 0.05f};
        
        v3 Color = {};
        // Check if hovered or clicked.
        {
            v2 Pos = V2S32(Input->MouseX, Input->MouseY);
            v2 Dim = V2S32(Buffer->Width, Buffer->Height);
            v2 ButtonMin = V2MulV2(Min, Dim); 
            v2 ButtonMax = V2MulV2(Max, Dim);
            
            b32 Hovered = false;
            b32 Clicked = false;
            Hovered = !!InBounds(Pos, ButtonMin, ButtonMax);;
            Clicked = !!(Hovered && Input->Buttons[PlatformButton_Left].EndedDown);
            
            Color = ((Clicked) ? (v3){HexToRGBV3(Color_ButtonPressed)} :
                     (Hovered) ? (v3){HexToRGBV3(Color_ButtonHovered)} :
                     (v3){HexToRGBV3(Color_Button)});
            
            if(Clicked) 
            {
                Animate = false;
                ResetApp(App);
            }
        }
        
        // Draw the button
        {
            // Prepare a quad
            s32 Count = 6;
            
            v2 ClipMin = V2MulV2(V2AddF32(V2MulF32(Min, 2.0f), -1.0f), (v2){1.0f, -1.0f});
            v2 ClipMax = V2MulV2(V2AddF32(V2MulF32(Max, 2.0f), -1.0f), (v2){1.0f, -1.0f});
            
            Vertices[0] = {ClipMin.X, ClipMin.Y, -1.0f}; // BL
            Vertices[1] = {ClipMax.X, ClipMin.Y, -1.0f}; // BR
            Vertices[2] = {ClipMin.X, ClipMax.Y, -1.0f}; // TL
            Vertices[3] = {ClipMin.X, ClipMax.Y, -1.0f}; // TL
            Vertices[4] = {ClipMax.X, ClipMax.Y, -1.0f}; // TR
            Vertices[5] = {ClipMax.X, ClipMin.Y, -1.0f}; // BR
            for EachIndex(Idx, Count) Vertices[Idx].Z = -1.0f;
            
            ButtonShader = ProgramFromShaders(FrameArena, App,
                                              S8(Strs_ButtonVertexShaderPath), S8(Strs_ButtonFragmentShaderPath));
            glUseProgram(ButtonShader);
            
            {            
                glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
                
                PosAttrib = glGetAttribLocation(ButtonShader, "pos");
                glEnableVertexAttribArray(PosAttrib);
                glVertexAttribPointer(PosAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(v3), 0);
                
                glBufferData(GL_ARRAY_BUFFER, sizeof(v3)*Count, Vertices, GL_STATIC_DRAW);
            }
            
            gl_handle UColor = glGetUniformLocation(ButtonShader, "color");
            gl_handle UButtonMin = glGetUniformLocation(ButtonShader, "buttonMin");
            gl_handle UButtonMax = glGetUniformLocation(ButtonShader, "buttonMax");
            
            glUniform2f(UButtonMin, Min.X, Max.Y);
            glUniform2f(UButtonMax, Max.X, Min.Y);
            glUniform3f(UColor, Color.R, Color.G, Color.B);
            
            glDrawArrays(GL_TRIANGLES, 0, Count);
        }
        
        // Draw the text
        {
            glEnable(GL_BLEND);  
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
            glDisable(GL_DEPTH_TEST);
            
            TextShader = ProgramFromShaders(FrameArena, App, 
                                            S8(Strs_TextVertexShaderPath),
                                            S8(Strs_TextFragmentShaderPath));
            glUseProgram(TextShader);
            
            f32 HeightPx = (f32)(TextImage.Width/30);
            
            local_persist app_font Font = {};
            if(!Font.Initialized)
            {
                char *FontPath = PathFromExe(FrameArena, App, S8("../data/font.ttf"));
                rlf_InitFont(&Font, FontPath);
            }
            
            f32 FontScale = stbtt_ScaleForPixelHeight(&Font.Info, HeightPx);
            f32 Baseline = rlf_GetBaseLine(&Font, FontScale);
            
            f32 X = (Min.X * (f32)TextImage.Width);
            f32 Y = (Min.Y * (f32)TextImage.Height);
            rlf_DrawTextFormat(FrameArena, &TextImage, &Font, 
                               HeightPx, {X, Y + Baseline}, Color_ButtonText, false,
                               " Reset");
            
            LoadTextureFromImage(Tex[1], TextImage.Width, TextImage.Height, TextImage.Pixels, GL_BGRA, TextShader);
            
            s32 Count = 6;
            Vertices[0] = {-1.0f, 1.0f, 0.0f}; // TL
            Vertices[1] = {1.0f, 1.0f, 0.0f}; // TR
            Vertices[2] = {-1.0f, -1.0f, 0.0f}; // BL
            Vertices[3] = {-1.0f, -1.0f, 0.0f}; // BL
            Vertices[4] = {1.0f, -1.0f, 0.0f}; // BR
            Vertices[5] = {1.0f, 1.0f, 0.0f}; // TR
            for EachIndex(Idx, Count) 
            {
                Vertices[Idx].Z = -1.0f;
                TexCoords[Idx].X = (Vertices[Idx].X + 1.0f) * 0.5f;
                TexCoords[Idx].Y = (1.0f - (Vertices[Idx].Y + 1.0f) * 0.5f);
            }
            
            glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(v3)*Count, Vertices, GL_STATIC_DRAW);
            
            PosAttrib = glGetAttribLocation(ShaderProgram, "pos");
            glEnableVertexAttribArray(PosAttrib);
            glVertexAttribPointer(PosAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(v3), 0);
            
            glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(v2)*Count, TexCoords, GL_STATIC_DRAW);
            
            TexAttrib = glGetAttribLocation(TextShader, "tex");
            glEnableVertexAttribArray(TexAttrib);
            glVertexAttribPointer(TexAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(v2), 0);
            
            glDrawArrays(GL_TRIANGLES, 0, Count);
        }
    }
    
    // Cleanup
    {    
        glDeleteBuffers(2, &VBO[0]);
        glDeleteVertexArrays(1, &VAO);
        glDeleteTextures(2, &Tex[0]);
        glDeleteProgram(ShaderProgram);
        glDeleteProgram(TextShader);
        glDeleteProgram(ButtonShader);
    }
    
    OS_ProfileAndPrint("GL");
    
    return ShouldQuit;
}