#include "base/base.h"
#include "ex_platform.h"

#define GLAD_GL_IMPLEMENTATION
#include "lib/gl_core.h"

typedef unsigned int gl_handle;

typedef struct vertex vertex;
struct vertex
{
    f32 X;
    f32 Y;
    f32 Z;
};

typedef struct tex_coord tex_coord;
struct tex_coord
{
    f32 X;
    f32 Y;
};

typedef vertex normal;

typedef s32 face[4][3];

#define New(type, Name, Array, Count) type *Name = Array + Count; Count += 1;

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

#define HexToRGBV3(Hex) \
((f32)((Hex >> 8*2) & 0xFF)/255.0f), \
((f32)((Hex >> 8*1) & 0xFF)/255.0f), \
((f32)((Hex >> 8*0) & 0xFF)/255.0f)

//~ Math
internal vertex 
Rotate(vertex V, f32 Angle)
{
    vertex Result = {};
    
    f32 C = cosf(Angle);
    f32 S = sinf(Angle);
    
    f32 X = V.X;
    f32 Y = V.Y;
    f32 Z = V.Z;
    
    Result.X = X*C - Z*S;
    Result.Y = Y;
    Result.Z = X*S + Z*C;
    
    return Result;
}

//~ GLAD
void GLADNullPreCallback(const char *name, GLADapiproc apiproc, int len_args, ...) {}

void GLADNullPostCallback(void *ret, const char *name, GLADapiproc apiproc, int len_args, ...) {}

void GLADDisableCallbacks()
{
    _pre_call_gl_callback = GLADNullPreCallback;
    _post_call_gl_callback = GLADNullPostCallback;
}

void GLADEnableCallbacks()
{
    _pre_call_gl_callback = _pre_call_gl_callback_default;
    _post_call_gl_callback = _post_call_gl_callback_default;
}


//~ Helpers
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
    }
}

internal gl_handle
CompileShaderFromSource(str8 Source, s32 Type)
{
    gl_handle Handle = glCreateShader(Type);
    
    glShaderSource(Handle, 1, (char **)&Source.Data, NULL);
    glCompileShader(Handle);
    GLErrorStatus(Handle, true);
    
    return Handle;
}


//~ Parsing

C_LINKAGE 
UPDATE_AND_RENDER(UpdateAndRender)
{
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
            
            App->Initialized = true;
        }
        
#if !RL_INTERNAL    
        GLADDisableCallbacks();
#endif
    }
    OS_ProfileAndPrint("Init");
    
    //Input 
    for EachIndex(Idx, Input->Text.Count)
    {
        app_text_button Key = Input->Text.Buffer[Idx];
        
        if(Key.Codepoint == 'a') App->XOffset = 0.0f;
    }
    
    s32 Major, Minor;
    glGetIntegerv(GL_MAJOR_VERSION, &Major);
    glGetIntegerv(GL_MINOR_VERSION, &Minor);
    
    glViewport(0, 0, Buffer->Width, Buffer->Height);
    
    str8 VertexShaderSource = OS_ReadEntireFileIntoMemory("./code/example/vert.glsl");
    str8 FragmentShaderSource = OS_ReadEntireFileIntoMemory("./code/example/frag.glsl");
    gl_handle VertexShader = CompileShaderFromSource(VertexShaderSource, GL_VERTEX_SHADER);
    gl_handle FragmentShader = CompileShaderFromSource(FragmentShaderSource, GL_FRAGMENT_SHADER);
    gl_handle ShaderProgram = glCreateProgram();
    glAttachShader(ShaderProgram, VertexShader);
    glAttachShader(ShaderProgram, FragmentShader);
    glLinkProgram(ShaderProgram);
    GLErrorStatus(ShaderProgram, false);
    
    glDeleteShader(FragmentShader); 
    glDeleteShader(VertexShader);
    OS_FreeFileMemory(VertexShaderSource);
    OS_FreeFileMemory(FragmentShaderSource);
    
    glUseProgram(ShaderProgram);
    
    gl_handle VAO, VBO;
    glGenVertexArrays(1, &VAO); 
    glGenBuffers(1, &VBO);  
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    vertex Vertices[] = 
    {
#if 1
        // Cube
        {0.5f,  -0.5f,  0.5f}, // BR
        {-0.5f, -0.5f,  0.5f}, // BL
        {-0.5f,  0.5f,  0.5f}, // TL
        {0.5f,  0.5f,   0.5f}, // TR
        
        {0.5f,  -0.5f,  -0.5f},
        {-0.5f, -0.5f,  -0.5f},
        {-0.5f,  0.5f,  -0.5f}, 
        {0.5f,  0.5f,   -0.5f},
#elif 1
        // Rectangle out of two triangles
        {0.5f,  -0.5f,  0.5f},
        {-0.5f, -0.5f,  0.5f},
        {-0.5f,  0.5f,  0.5f}, 
        
        {0.5f,  -0.5f,  0.5f},
        {-0.5f, 0.5f,  0.5f},
        {0.5f,  0.5f,  0.5f}, 
#endif
    };
    
    GLuint Faces[] =
    {
        0, 1, 2, 3,
        4, 5, 6, 7,
        1, 2, 6, 5,
        0, 3, 7, 4,
    };
    
    App->XOffset += Input->dtForFrame;
    
    GLint PosAttrib = glGetAttribLocation(ShaderProgram, "pos");
    glEnableVertexAttribArray(PosAttrib);
    glVertexAttribPointer(PosAttrib, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
    
    f32 Angle = Pi32 * App->XOffset;
    vertex Color = {cosf(Angle*2.0f), 0.5f, 0.0f};
    
    Log("Offset: %.4f\n", Angle);
    
    GLint UAngle = glGetUniformLocation(ShaderProgram, "angle");
    GLint UColor = glGetUniformLocation(ShaderProgram, "color");
    glUniform2f(UAngle, Angle, 0.25f*Pi32);
    glUniform3f(UColor, Color.X, Color.Y, Color.Z);
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW);
    
    b32 Fill = true;
    s32 Mode = (Fill) ? GL_FILL : GL_LINE;
    glPolygonMode(GL_FRONT_AND_BACK, Mode);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(HexToRGBV3(ColorMapBackground), 0.0f);
    glLineWidth(2.0f);
    
    for EachIndex(Idx, 4)
    {
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, Faces + 4*Idx);
    }
    
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(ShaderProgram);
    
    OS_ProfileAndPrint("GL");
    
    return false;
}