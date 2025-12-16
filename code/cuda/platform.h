/* date = December 14th 2025 5:30 pm */

#ifndef PLATFORM_H
#define PLATFORM_H

struct app_offscreen_buffer
{
    s32 Width;
    s32 Height;
    u8 *Pixels;
    s32 Pitch;
    s32 BytesPerPixel;
};

struct app_text_button
{
    rune Codepoint;
    // TODO(luca): Use flag and bits.
    b32 Control;
    b32 Shift;
    b32 Alt;
};
typedef struct app_text_button app_text_button;

struct app_button_state
{
    s32 HalfTransitionCount;
    b32 EndedDown;
};
typedef struct app_button_state app_button_state;

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

inline b32 WasPressed(app_button_state State)
{
    b32 Result = ((State.HalfTransitionCount > 1) || 
                  (State.HalfTransitionCount == 1 && State.EndedDown));
    return Result;
}

//~ App logic
struct point
{
    f32 Lat;
    f32 Lon;
};


#include "lib/handmade_font.h"

struct random_series
{
    u64 State;
    u64 Increment;
    u64 Multiplier;
};
typedef struct random_series random_series;

typedef struct app_state app_state;
struct app_state
{
    point *Points;
    u32 PointsCount;
    u32 MaxPointsCount;
    
    arena *PermanentCPUArena;
    arena *PermanentGPUArena;
    
    s32 GenerateAmount;
    
    app_font Font;
    
    random_series Series;
    
    b32 Initialized;
};

//~ Functions
#define UPDATE_AND_RENDER(Name) void Name(thread_context *Context, app_state *App, arena *CPUFrameArena, arena *GPUFrameArena, app_offscreen_buffer *Buffer, app_input *Input)
typedef UPDATE_AND_RENDER(update_and_render);




#endif //PLATFORM_H
