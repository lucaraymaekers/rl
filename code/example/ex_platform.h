/* date = December 14th 2025 5:30 pm */

#ifndef PLATFORM_H
#define PLATFORM_H

#include "ex_random.h"

struct app_offscreen_buffer
{
    s32 Width;
    s32 Height;
    u8 *Pixels;
    s32 Pitch;
    s32 BytesPerPixel;
};

enum platform_key
{
    PlatformKey_None = 0,
    PlatformKey_Tab,
    PlatformKey_Return,
    PlatformKey_Escape,
    
    PlatformKey_Delete,
    PlatformKey_BackSpace,
    PlatformKey_Insert,
    
    PlatformKey_F1,
    PlatformKey_F2,
    PlatformKey_F3,
    PlatformKey_F4,
    PlatformKey_F5,
    PlatformKey_F6,
    PlatformKey_F7,
    PlatformKey_F8,
    PlatformKey_F9,
    PlatformKey_F10,
    PlatformKey_F11,
    PlatformKey_F12,
    
    PlatformKey_Home,
    PlatformKey_End,
    PlatformKey_PageUp,
    PlatformKey_PageDown,
    
    PlatformKey_Up,
    PlatformKey_Down,
    PlatformKey_Left,
    PlatformKey_Right,
    
    PlatformKey_Shift,
    PlatformKey_Control,
    PlatformKey_Alt,
    
    PlatformKey_Count,
};
typedef enum platform_key platform_key;

enum platform_key_modifier
{
    PlatformKeyModifier_None    = 0,
    PlatformKeyModifier_Shift   = (1 << 0),
    PlatformKeyModifier_Control = (1 << 1),
    PlatformKeyModifier_Alt     = (1 << 2),
    PlatformKeyModifier_Any     = (1 << 3),
};
typedef enum platform_key_modifier platform_key_modifier;

typedef struct app_text_button app_text_button;
struct app_text_button
{
    union
    {
        rune Codepoint;
        platform_key Symbol;
    };
    s32 Modifiers;
    b32 IsSymbol;
};

typedef struct app_button_state app_button_state;
struct app_button_state
{
    s32 HalfTransitionCount;
    b32 EndedDown;
};

enum platform_cursor_shape
{
    PlatformCursorShape_None = 0,
    PlatformCursorShape_Grab,
};
typedef enum platform_cursor_shape platform_cursor_shape;

enum platform_mouse_buttons
{
    PlatformButton_Left = 0,
    PlatformButton_Right,
    PlatformButton_Middle,
    PlatformButton_ScrollUp,
    PlatformButton_ScrollDown,
    PlatformButton_Count
};
typedef enum platform_mouse_buttons platform_mouse_buttons;

typedef struct app_input app_input;
struct app_input
{
    app_button_state Buttons[PlatformButton_Count];
    s32 MouseX, MouseY, MouseZ;
    
    struct Text
    {
        u32 Count;
        app_text_button Buffer[64];
    } Text;
    
    f32 dtForFrame;
};

NO_WARNINGS_BEGIN
#include "ex_font.h"
NO_WARNINGS_END

typedef struct model_path model_path;
struct model_path
{
    str8 Model;
    str8 Texture;
};

//~ App logic
typedef struct app_state app_state;
struct app_state
{
    arena *PermanentArena;
    random_series Series;
    
    app_font Font;
    
    // App things
    v3 Offset;
    v2 Angle;
    b32 Animate;
    
    model_path CurrentModel;
    
    str8 ExeDirPath;
#if RL_INTERNAL
    b32 DebuggerAttached;
#endif
    b32 Reloaded;
    b32 Initialized;
};

//- Helper functions 
internal inline b32 
WasPressed(app_button_state State)
{
    b32 Result = ((State.HalfTransitionCount > 1) || 
                  (State.HalfTransitionCount == 1 && State.EndedDown));
    return Result;
}

internal inline b32
CharPressed(app_input *Input, rune Codepoint, s32 Modifiers)
{
    b32 Pressed = false;
    for EachIndex(Idx, Input->Text.Count)
    {
        app_text_button Key = Input->Text.Buffer[Idx];
        b32 ModifiersMatch = ((Modifiers == PlatformKeyModifier_Any) ||
                              (Key.Modifiers == Modifiers));
        
        if((Key.Modifiers & PlatformKeyModifier_Shift) && 
           (Key.Codepoint >= L'A' && Key.Codepoint <= L'Z'))
        {
            Key.Codepoint += 32;
        }
        
        if(Codepoint >= L'A' && Codepoint <= L'Z') Codepoint += 32;
        
        
        if(Key.Codepoint == Codepoint && ModifiersMatch)
        {
            Pressed = true;
            break;
        }
    }
    return Pressed;
}

internal char *
PathFromExe(arena *Arena, app_state *App, str8 Path)
{
    char *Result = 0;
    
    str8 Base = App->ExeDirPath;
    umm Size = Base.Size + Path.Size + 1;
    
    Result = PushArray(Arena, char, Size);
    
    umm At = 0;
    for EachIndex(Idx, Base.Size)
    {
        Result[At] = Base.Data[Idx];
        At += 1;
    }
    for EachIndex(Idx, Path.Size)
    {
        Result[At] = Path.Data[Idx];
        At += 1;
    }
    
    Result[Size - 1] = 0;
    
    return Result;
}

//~ Functions
#define UPDATE_AND_RENDER(Name) b32 Name(thread_context *Context, app_state *App, arena *FrameArena, app_offscreen_buffer *Buffer, app_input *Input)
typedef UPDATE_AND_RENDER(update_and_render);

UPDATE_AND_RENDER(UpdateAndRenderStub) { return false; }

typedef struct app_code app_code;
struct app_code
{
    update_and_render *UpdateAndRender;
    
    char *LibraryPath;
    umm LibraryHandle;
    b32 Loaded;
};

//~ API
typedef umm P_context;

internal P_context P_ContextInit(arena *Arena, app_offscreen_buffer *Buffer, b32 *Running);
internal void      P_UpdateImage(P_context Context, app_offscreen_buffer *Buffer);
internal void      P_ProcessMessages(P_context Context, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running);
internal void      P_LoadAppCode(arena *Arena, app_code *Code, app_state *App, s64 *LastWriteTime);
#endif //PLATFORM_H
