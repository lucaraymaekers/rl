#include "base/base.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>
#include <X11/extensions/Xrandr.h>
#include <X11/cursorfont.h>
#include <signal.h>

struct linux_x11_context
{
    XImage *WindowImage;
    Display *DisplayHandle;
    Window WindowHandle;
    GC DefaultGC;
    XIC InputContext;
    Atom WM_DELETE_WINDOW;
    b32 Initialized;
};

global_variable b32 *GlobalRunning;

internal rune
ConvertUTF8StringToRune(u8 UTF8String[4])
{
    rune Codepoint = 0;
    
    if((UTF8String[0] & 0x80) == 0x00)
    {
        Codepoint = UTF8String[0];
    }
    else if((UTF8String[0] & 0xE0) == 0xC0)
    {
        Codepoint = (
                     ((UTF8String[0] & 0x1F) << 6*1) |
                     ((UTF8String[1] & 0x3F) << 6*0)
                     );
    }
    else if((UTF8String[0] & 0xF0) == 0xE0)
    {
        Codepoint = (
                     ((UTF8String[0] & 0x0F) << 6*2) |
                     ((UTF8String[1] & 0x3F) << 6*1) |
                     ((UTF8String[2] & 0x3F) << 6*0)
                     );
    }
    else if((UTF8String[0] & 0xF8) == 0xF8)
    {
        Codepoint = (
                     ((UTF8String[0] & 0x0E) << 6*3) |
                     ((UTF8String[1] & 0x3F) << 6*2) |
                     ((UTF8String[2] & 0x3F) << 6*1) |
                     ((UTF8String[3] & 0x3F) << 6*0)
                     );
    }
    else
    {
        Assert(0);
    }
    
    return Codepoint;
}

internal void
LinuxSIGINTHandler(int SigNum)
{
    *GlobalRunning = false;
}

internal void
LinuxSetSIGINT(b32 *Running)
{
    GlobalRunning = Running; 
    signal(SIGINT, LinuxSIGINTHandler);
}


internal void 
LinuxProcessKeyPress(app_button_state *ButtonState, b32 IsDown)
{
    if(ButtonState->EndedDown != IsDown)
    {
        ButtonState->EndedDown = IsDown;
        ButtonState->HalfTransitionCount++;
    }
}

//~ Platform API
internal s64 
P_GetWallClock()
{
    s64 Result = 0;
    
    struct timespec Counter;
    clock_gettime(CLOCK_MONOTONIC, &Counter);
    Result = (s64)Counter.tv_sec*1000000000 + (s64)Counter.tv_nsec;
    
    return Result;
}

internal P_context
P_ContextInit(arena *Arena, app_offscreen_buffer *Buffer, b32 *Running)
{
    P_context Result = 0;
    
    linux_x11_context *Context = PushStruct(Arena, linux_x11_context);
    
    LinuxSetSIGINT(Running);
    
    s32 XRet = 0;
    
    Context->DisplayHandle = XOpenDisplay(0);
    if(Context->DisplayHandle)
    {
        Window RootWindow = XDefaultRootWindow(Context->DisplayHandle);
        s32 Screen = XDefaultScreen(Context->DisplayHandle);
        s32 ScreenBitDepth = 24;
        XVisualInfo WindowVisualInfo = {};
        if(XMatchVisualInfo(Context->DisplayHandle, Screen, ScreenBitDepth, TrueColor, &WindowVisualInfo))
        {
            XSetWindowAttributes WindowAttributes = {};
            WindowAttributes.bit_gravity = StaticGravity;
#if HANDMADE_INTERNAL            
            WindowAttributes.background_pixel = 0xFF00FF;
#endif
            WindowAttributes.colormap = XCreateColormap(Context->DisplayHandle, RootWindow, WindowVisualInfo.visual, AllocNone);
            WindowAttributes.event_mask = (StructureNotifyMask | 
                                           KeyPressMask | KeyReleaseMask | 
                                           ButtonPressMask | ButtonReleaseMask |
                                           EnterWindowMask | LeaveWindowMask);
            u64 WindowAttributeMask = CWBitGravity | CWBackPixel | CWColormap | CWEventMask;
            
            Context->WindowHandle = XCreateWindow(Context->DisplayHandle, RootWindow,
                                                  0, 0,
                                                  Buffer->Width, Buffer->Height,
                                                  0,
                                                  WindowVisualInfo.depth, InputOutput,
                                                  WindowVisualInfo.visual, WindowAttributeMask, &WindowAttributes);
            if(Context->WindowHandle)
            {
                XRet = XStoreName(Context->DisplayHandle, Context->WindowHandle, "Handmade Window");
                
                // NOTE(luca): If we set the MaxWidth and MaxHeigth to the same values as MinWidth and MinHeight there is a bug on dwm where it won't update the window decorations when trying to remove them.
                // In the future we will allow resizing to any size so this does not matter that much.
#if 0                    
                LinuxSetSizeHint(Context->DisplayHandle, Context->WindowHandle, 0, 0, 0, 0);
#endif
                
                // NOTE(luca): Tiling window managers should treat windows with the WM_TRANSIENT_FOR property as pop-up windows.  This way we ensure that we will be a floating window.  This works on my setup (dwm). 
                XRet = XSetTransientForHint(Context->DisplayHandle, Context->WindowHandle, RootWindow);
                
                XClassHint ClassHint = {};
                ClassHint.res_name = "Handmade Window";
                ClassHint.res_class = "Handmade Window";
                XSetClassHint(Context->DisplayHandle, Context->WindowHandle, &ClassHint);
                
                XSetLocaleModifiers("");
                
                XIM InputMethod = XOpenIM(Context->DisplayHandle, 0, 0, 0);
                if(!InputMethod){
                    XSetLocaleModifiers("@im=none");
                    InputMethod = XOpenIM(Context->DisplayHandle, 0, 0, 0);
                }
                Context->InputContext = XCreateIC(InputMethod,
                                                  XNInputStyle, XIMPreeditNothing | XIMStatusNothing,
                                                  XNClientWindow, Context->WindowHandle,
                                                  XNFocusWindow,  Context->WindowHandle,
                                                  NULL);
                XSetICFocus(Context->InputContext);
                
                s32 BitsPerPixel = Buffer->BytesPerPixel*8;
                Context->WindowImage = XCreateImage(Context->DisplayHandle, WindowVisualInfo.visual, WindowVisualInfo.depth, ZPixmap, 0, (char *)Buffer->Pixels, Buffer->Width, Buffer->Height, BitsPerPixel, 0);
                Context->DefaultGC = DefaultGC(Context->DisplayHandle, Screen);
                
                XRet = XMapWindow(Context->DisplayHandle, Context->WindowHandle);
                XRet = XFlush(Context->DisplayHandle);
                
                Context->WM_DELETE_WINDOW = XInternAtom(Context->DisplayHandle, "WM_DELETE_WINDOW", False);
                XRet = XSetWMProtocols(Context->DisplayHandle, Context->WindowHandle, 
                                       &Context->WM_DELETE_WINDOW, 1);
                Assert(XRet);
                Context->Initialized = true;
                
                Result = (umm)Context;
            }
        }
    }
    
    return Result;
}

internal void 
P_ProcessMessages(P_context Context, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running)
{
    linux_x11_context *Linux = (linux_x11_context *)Context;
    
	if(Linux)
	{
        XEvent WindowEvent = {};
        while(XPending(Linux->DisplayHandle) > 0)
        {
            XNextEvent(Linux->DisplayHandle, &WindowEvent);
            b32 FilteredEvent = XFilterEvent(&WindowEvent, 0);
            if(FilteredEvent)
            {
                // TODO(luca): Logging
                // NOTE(luca): I really don't know what I should expect here.
            }
            
            switch(WindowEvent.type)
            {
                case KeyPress:
                case KeyRelease:
                {
                    //- How text input works 
                    // The needs:
                    //  1. Preserve game buttons, so that we can switch between a "game mode" or 
                    //     "text input mode".
                    //  2. Text input using the input method of the user which should allow for utf8 characters.
                    //  3. Hotkey support.  Eg. quickly navigating text.
                    // 3 will be supported by 2 for code reuse.
                    //
                    // We are going to send a buffer text button presses to the game layer, this solves these
                    // issues:
                    // - Pressing the same key multiple times in one frame.
                    // - Having modifiers be specific to each key press.
                    // - Not having to create a button record for each possible character in the structure.
                    // - Key events come in one at a time in the event loop, thus we need to have a buffer for
                    //   multiple keys pressed on a single frame.
                    //
                    // We store a count along the buffer and in the buffer we store the utf8 codepoint and its
                    // modifiers.
                    // The app code is responsible for traversing this buffer and applying the logic. 
                    
                    // The problem of input methods and hotkeys: 
                    // Basically the problem is that if we allow the input method and combo's that could be 
                    // filtered by the input method it won't seem consistent to the user.
                    // So we don't allow key bound to the input method to have an effect and we only pass key
                    // inputs that have not been filtered.
                    //
                    // In the platform layer we handle the special case were the input methods creates non-
                    // printable characters and we decompose those key inputs since non-printable characters
                    // have no use anymore.
                    
                    // Extra:
                    // - I refuse to check which keys bind to what modifiers. It's not important.
                    
                    // - Handy resources: 
                    //   - https://www.coderstool.com/unicode-text-converter
                    //   - man Compose(5).
                    //   - https://en.wikipedia.org/wiki/Control_key#History
                    
                    KeySym Symbol = XLookupKeysym(&WindowEvent.xkey, 0);
                    b32 IsDown = (WindowEvent.type == KeyPress);
                    
                    // TODO(luca): Refresh mappings.
                    // NOTE(luca): Only KeyPress events  see man page of Xutf8LookupString().  And skip filtered events for text input, but keep them for controller.
                    if(IsDown && !FilteredEvent)
                    {
                        // TODO(luca): Choose a better error value.
                        rune Codepoint = L'ù';
                        u8 LookupBuffer[4] = {};
                        Status LookupStatus = {};
                        
                        s32 BytesLookepdUp = Xutf8LookupString(Linux->InputContext, &WindowEvent.xkey, 
                                                               (char *)&LookupBuffer, ArrayCount(LookupBuffer), 
                                                               0, &LookupStatus);
                        Assert(LookupStatus != XBufferOverflow);
                        Assert(BytesLookepdUp <= 4);
                        
                        if(LookupStatus!= XLookupNone &&
                           LookupStatus!= XLookupKeySym)
                        {
                            if(BytesLookepdUp)
                            {
                                Assert(Input->Text.Count < ArrayCount(Input->Text.Buffer));
                                
                                Codepoint = ConvertUTF8StringToRune(LookupBuffer);
                                
                                // NOTE(luca): Input methods might produce non printable characters (< ' ').  If this
                                // happens we try to "decompose" the key input.
                                if(Codepoint < ' ' && Codepoint >= 0)
                                {
                                    if(Symbol >= XK_space)
                                    {
                                        Codepoint = (char)(' ' + (Symbol - XK_space));
                                    }
                                }
                                
                                // NOTE(luca): Since this is only for text input we pass Return and Backspace as codepoints.
                                if((Codepoint >= ' ' || Codepoint < 0) ||
                                   Codepoint == '\r' || Codepoint == '\b' || Codepoint == '\n')
                                {                            
                                    app_text_button *TextButton = &Input->Text.Buffer[Input->Text.Count++];
                                    TextButton->Codepoint = Codepoint;
                                    TextButton->Shift   = (WindowEvent.xkey.state & ShiftMask);
                                    TextButton->Control = (WindowEvent.xkey.state & ControlMask);
                                    TextButton->Alt     = (WindowEvent.xkey.state & Mod1Mask);
#if 0                           
                                    printf("%d bytes '%c' %d (%c|%c|%c)\n", 
                                           BytesLookepdUp, 
                                           ((Codepoint >= ' ') ? (char)Codepoint : '\0'),
                                           Codepoint,
                                           ((WindowEvent.xkey.state & ShiftMask)   ? 'S' : ' '),
                                           ((WindowEvent.xkey.state & ControlMask) ? 'C' : ' '),
                                           ((WindowEvent.xkey.state & Mod1Mask)    ? 'A' : ' '));
#endif
                                }
                                else
                                {
                                    // TODO(luca): Logging
                                }
                                
                            }
                        }
                    }
                    else if((WindowEvent.xkey.state & Mod1Mask) && 
                            (Symbol == XK_F4))
                    {
                        *Running = false;
                    }
                } break;
                
                case ButtonPress:
                case ButtonRelease:
                {
                    b32 IsDown = (WindowEvent.type == ButtonPress);
                    u32 Button = WindowEvent.xbutton.button;
                    
                    if(0) {}
                    else if(Button == Button1)
                    {
                        LinuxProcessKeyPress(&Input->Buttons[PlatformButton_Left], IsDown);
                    }
                    else if(Button == Button2)
                    {
                        LinuxProcessKeyPress(&Input->Buttons[PlatformButton_Middle], IsDown);
                    }
                    else if(Button == Button3)
                    {
                        LinuxProcessKeyPress(&Input->Buttons[PlatformButton_Right], IsDown);
                    }
                    else if(Button == Button4)
                    {
                        LinuxProcessKeyPress(&Input->Buttons[PlatformButton_ScrollUp], IsDown);
                    }
                    else if(Button == Button5)
                    {
                        LinuxProcessKeyPress(&Input->Buttons[PlatformButton_ScrollDown], IsDown);
                    }
                } break;
                
                case DestroyNotify:
                {
                    XDestroyWindowEvent *Event = (XDestroyWindowEvent *)&WindowEvent;
                    if(Event->window == Linux->WindowHandle)
                    {
                        *Running = false;
                    }
                } break;
                
                case ClientMessage:
                {
                    XClientMessageEvent *Event = (XClientMessageEvent *)&WindowEvent;
                    if((Atom)Event->data.l[0] == Linux->WM_DELETE_WINDOW)
                    {
                        XDestroyWindow(Linux->DisplayHandle, Linux->WindowHandle);
                        *Running = false;
                    }
                } break;
                
                case EnterNotify:
                {
                    //LinuxHideCursor(DisplayHandle, WindowHandle);
                } break;
                
                case LeaveNotify:
                {
                    //LinuxShowCursor(DisplayHandle, WindowHandle);
                } break;
                
            }
        }
        
        // Window could have been closed
        if(*Running)
        {        
            // TODO(luca): Move this into process pending messages.
            s32 MouseX = 0, MouseY = 0, MouseZ = 0;
            u32 MouseMask = 0;
            u64 Ignored;
            if(XQueryPointer(Linux->DisplayHandle, Linux->WindowHandle, 
                             &Ignored, &Ignored, (int *)&Ignored, (int *)&Ignored,
                             &MouseX, &MouseY, &MouseMask))
            {
                if(MouseX <= Buffer->Width && MouseX >= 0 &&
                   MouseY <= Buffer->Height && MouseY >= 0)
                {
                    Input->MouseY = MouseY;
                    Input->MouseX = MouseX;
                }
            }
        }
	}
}

internal void
P_UpdateImage(P_context Context, app_offscreen_buffer *Buffer)
{
    linux_x11_context *Linux = (linux_x11_context *)Context;
	if(Linux)
	{
        XPutImage(Linux->DisplayHandle,
                  Linux->WindowHandle, 
                  Linux->DefaultGC, 
                  Linux->WindowImage, 0, 0, 0, 0, 
                  Buffer->Width, 
                  Buffer->Height);
	}
}