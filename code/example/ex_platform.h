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

typedef struct app_text_button app_text_button;
struct app_text_button
{
    rune Codepoint;
    // TODO(luca): Use flag and bits.
    b32 Control;
    b32 Shift;
    b32 Alt;
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
CharPressed(app_input *Input, rune Codepoint)
{
    b32 Pressed = false;
    for EachIndex(Idx, Input->Text.Count)
    {
        if(Input->Text.Buffer[Idx].Codepoint == Codepoint)
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
    
#if RL_INTERNAL
    b32 DebuggerAttached;
#endif
    
    b32 Initialized;
};

//~ Functions
#define UPDATE_AND_RENDER(Name) void Name(thread_context *Context, app_state *App, arena *FrameArena, app_offscreen_buffer *Buffer, app_input *Input)
typedef UPDATE_AND_RENDER(update_and_render);


//~ API
typedef umm P_context;

internal P_context P_ContextInit(arena *Arena, app_offscreen_buffer *Buffer, b32 *Running);
internal void      P_UpdateImage(P_context Context, app_offscreen_buffer *Buffer);
internal void      P_ProcessMessages(P_context Context, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running);

//- Helpers 
internal inline f32
P_SecondsElapsed(s64 Start, s64 End)
{
    f32 Result = ((f32)(End - Start)/1000000000.0f);
    return Result;
}

internal inline f32
P_MSElapsed(s64 Start, s64 End)
{
    f32 Result = ((f32)(End - Start)/1000000.0f);
    return Result;
}

#endif //PLATFORM_H
