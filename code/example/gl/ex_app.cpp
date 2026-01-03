#include "base/base.h"
#include "ex_platform.h"

#define GLAD_GL_IMPLEMENTATION
#include "lib/gl_core.h"

NO_WARNINGS_BEGIN
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/handmade_font.h"
NO_WARNINGS_END


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
GLErrorStatus(GLuint Handle, b32 IsShader)
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

internal GLuint
CompileShaderFromSource(str8 Source, s32 Type)
{
    GLuint Handle = glCreateShader(Type);
    
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
    
    local_persist b32 Animate = false;
    
    //Input 
    for EachIndex(Idx, Input->Text.Count)
    {
        app_text_button Key = Input->Text.Buffer[Idx];
        
        if(Key.Codepoint == 'r') App->XOffset = 0.0f;
        if(Key.Codepoint == 'a') Animate = !Animate;
        if(!Animate && Key.Codepoint == 'i') App->XOffset += Input->dtForFrame;
    }
    
    if(Animate)
    {    
        App->XOffset += Input->dtForFrame;
    }
    
    s32 Major, Minor;
    glGetIntegerv(GL_MAJOR_VERSION, &Major);
    glGetIntegerv(GL_MINOR_VERSION, &Minor);
    
    glViewport(0, 0, Buffer->Width, Buffer->Height);
    
    GLfloat Vertices[] = {
        //Position    Color             Texcoords
        -0.5f,  0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // Top-left
        0.5f,   0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // Top-right
        0.5f,  -0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // Bottom-right
        -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f  // Bottom-left
    };
    
    GLuint Elements[] = 
    {
        0, 1, 2,
        2, 3, 0
    };
    
    GLuint VAO;
    {
        glGenVertexArrays(1, &VAO); 
        glBindVertexArray(VAO);
    }
    
    GLuint VBO;
    {
        glGenBuffers(1, &VBO);  
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
    }
    
    GLuint EBO;
    {    
        glGenBuffers(1, &EBO);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Elements), Elements, GL_STATIC_DRAW);
    }
    
    GLuint ShaderProgram;
    {    
        str8 VertexShaderSource = OS_ReadEntireFileIntoMemory("./code/example/gl/vert.glsl");
        str8 FragmentShaderSource = OS_ReadEntireFileIntoMemory("./code/example/gl/frag.glsl");
        GLuint VertexShader = CompileShaderFromSource(VertexShaderSource, GL_VERTEX_SHADER);
        GLuint FragmentShader = CompileShaderFromSource(FragmentShaderSource, GL_FRAGMENT_SHADER);
        ShaderProgram = glCreateProgram();
        glAttachShader(ShaderProgram, VertexShader);
        glAttachShader(ShaderProgram, FragmentShader);
        glBindFragDataLocation(ShaderProgram, 0, "FragColor");
        glLinkProgram(ShaderProgram);
        GLErrorStatus(ShaderProgram, false);
        
        glDeleteShader(FragmentShader); 
        glDeleteShader(VertexShader);
        OS_FreeFileMemory(VertexShaderSource);
        OS_FreeFileMemory(FragmentShaderSource);
        
        glUseProgram(ShaderProgram);
    }
    
    GLint PosAttrib; 
    {
        PosAttrib = glGetAttribLocation(ShaderProgram, "InPos");
        glEnableVertexAttribArray(PosAttrib);
        glVertexAttribPointer(PosAttrib, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), 0);
    }
    
    GLint ColAttrib;
    {
        ColAttrib = glGetAttribLocation(ShaderProgram, "InColor");
        glEnableVertexAttribArray(ColAttrib);
        glVertexAttribPointer(ColAttrib, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
    }
    
    GLint TexAttrib; 
    {
        TexAttrib = glGetAttribLocation(ShaderProgram, "InTexCoord");
        glEnableVertexAttribArray(TexAttrib);
        glVertexAttribPointer(TexAttrib, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (void*)(5 * sizeof(GLfloat)));
    }
    
    OS_ProfileAndPrint("GL init");
    
    GLuint Tex[2];
    {
        glGenTextures(2, Tex);
        
        // Load kitty texture
        {        
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, Tex[0]);
            
            // TODO(luca): Proper assets loading.
            {
                local_persist int Width, Height, Components;
                local_persist u8 *Image = 0;
                if(!Image) Image = stbi_load("./data/sample.png", &Width, &Height, &Components, 0);
                
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Image);
                glUniform1i(glGetUniformLocation(ShaderProgram, "TexKitten"), 0);
            }
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            // TODO(luca): Use mipmap
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        
        // Load puppy texture
        {        
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, Tex[1]);
            
            // TODO(luca): Proper assets loading.
            {
                local_persist int Width, Height, Components;
                local_persist u8 *Image = 0;
                if(!Image) Image = stbi_load("./data/sample2.png", &Width, &Height, &Components, 0);
                
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Image);
                glUniform1i(glGetUniformLocation(ShaderProgram, "TexPuppy"), 1);
            }
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            // TODO(luca): Use mipmap
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        
        f32 Color[] = { 1.0f, 0.0f, 0.0f, 1.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, Color);
    }
    
    OS_ProfileAndPrint("GL tex init");
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW);
    
    // Shader input
    {    
        f32 Time = Pi32 * App->XOffset;
        GLuint UTime = glGetUniformLocation(ShaderProgram, "Time");
        glUniform1f(UTime, Time);
    }
    
    // Draw
    {    
        b32 Fill = true;
        s32 Mode = (Fill) ? GL_FILL : GL_LINE;
        glPolygonMode(GL_FRONT_AND_BACK, Mode);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(HexToRGBV3(ColorMapBackground), 0.0f);
        glLineWidth(2.0f);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
    
    // Cleanup
    {    
        glDeleteBuffers(1, &VBO);
        glDeleteVertexArrays(1, &VAO);
        glDeleteTextures(2, Tex);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(ShaderProgram);
    }
    
    OS_ProfileAndPrint("GL");
    
    return false;
}
