/* date = December 14th 2025 5:30 pm */

#ifndef PLATFORM_H
#define PLATFORM_H

typedef struct game_text_button
{
    rune Codepoint;
    // TODO(luca): Use flag and bits.
    b32 Control;
    b32 Shift;
    b32 Alt;
} game_text_button;

typedef struct game_button_state
{
    s32 HalfTransitionCount;
    b32 EndedDown;
} game_button_state;

typedef enum
{
    PlatformCursorShape_None = 0,
    PlatformCursorShape_Grab,
} platform_cursor_shape;

typedef enum
{
    PlatformMouseButton_Left = 0,
    PlatformMouseButton_Right,
    PlatformMouseButton_Middle,
    PlatformMouseButton_ScrollUp,
    PlatformMouseButton_ScrollDown,
    PlatformMouseButton_Count
} platform_mouse_buttons;

typedef struct game_input
{
    game_button_state MouseButtons[PlatformMouseButton_Count];
    s32 MouseX, MouseY, MouseZ;
    
    struct
    {
        u32 Count;
        game_text_button Buffer[64];
    } Text;
    
    f32 dtForFrame;
} game_input;

inline b32 WasPressed(game_button_state State)
{
    b32 Result = ((State.HalfTransitionCount > 1) || 
                  (State.HalfTransitionCount == 1 && State.EndedDown));
    return Result;
}

#endif //PLATFORM_H
