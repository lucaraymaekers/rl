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
    
    PlatformKey_Count,
};
typedef enum platform_key platform_key;

enum platform_key_modifier
{
    PlatformKeyModifier_None    = 0,
    PlatformKeyModifier_Shift   = (1 << 0),
    PlatformKeyModifier_Control = (1 << 1),
    PlatformKeyModifier_Alt     = (1 << 2),
    PlatformKeyModifier_All     = (PlatformKeyModifier_Shift | 
                                   PlatformKeyModifier_Control |
                                   PlatformKeyModifier_Alt),
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
        if(Input->Text.Buffer[Idx].Codepoint == Codepoint && 
           Input->Text.Buffer[Idx].Modifiers & Modifiers)
        {
            Pressed = true;
            break;
        }
    }
    return Pressed;
}

//~ App logic
typedef struct app_state app_state;
struct app_state
{
    arena *PermanentArena;
    random_series Series;
    
    arena *NumbersArena;
    
    f32 XOffset;
    f32 YOffset;
    
#if RL_INTERNAL
    b32 DebuggerAttached;
    b32 Reloaded;
#endif
    
    b32 Initialized;
};

//~ Functions
#define UPDATE_AND_RENDER(Name) b32 Name(thread_context *Context, app_state *App, arena *FrameArena, app_offscreen_buffer *Buffer, app_input *Input)
typedef UPDATE_AND_RENDER(update_and_render);


//~ API
typedef umm P_context;

internal P_context P_ContextInit(arena *Arena, app_offscreen_buffer *Buffer, b32 *Running);
internal void      P_UpdateImage(P_context Context, app_offscreen_buffer *Buffer);
internal void      P_ProcessMessages(P_context Context, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running);
#endif //PLATFORM_H
