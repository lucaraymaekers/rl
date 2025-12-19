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

typedef struct timespec timespec;

internal timespec 
LinuxGetWallClock()
{
    timespec Counter = {};
    clock_gettime(CLOCK_MONOTONIC, &Counter);
    return Counter;
}

internal s64 
LinuxGetNSecondsElapsed(timespec Start, timespec End)
{
    s64 Result = 0;
    Result = ((s64)End.tv_sec*1000000000 + (s64)End.tv_nsec) - ((s64)Start.tv_sec*1000000000 + (s64)Start.tv_nsec);
    return Result;
}

internal f32 
LinuxGetSecondsElapsed(timespec Start, timespec End)
{
    f32 Result = 0;
    Result = ((f32)LinuxGetNSecondsElapsed(Start, End)/1000.0f/1000.0f/1000.0f);
    
    return Result;
}

linux_x11_context LinuxInitX11(app_offscreen_buffer *Buffer)
{
    linux_x11_context Result = {};
    
    s32 XRet = 0;
    
    Result.DisplayHandle = XOpenDisplay(0);
    if(Result.DisplayHandle)
    {
        Window RootWindow = XDefaultRootWindow(Result.DisplayHandle);
        s32 Screen = XDefaultScreen(Result.DisplayHandle);
        s32 ScreenBitDepth = 24;
        XVisualInfo WindowVisualInfo = {};
        if(XMatchVisualInfo(Result.DisplayHandle, Screen, ScreenBitDepth, TrueColor, &WindowVisualInfo))
        {
            XSetWindowAttributes WindowAttributes = {};
            WindowAttributes.bit_gravity = StaticGravity;
#if HANDMADE_INTERNAL            
            WindowAttributes.background_pixel = 0xFF00FF;
#endif
            WindowAttributes.colormap = XCreateColormap(Result.DisplayHandle, RootWindow, WindowVisualInfo.visual, AllocNone);
            WindowAttributes.event_mask = (StructureNotifyMask | 
                                           KeyPressMask | KeyReleaseMask | 
                                           ButtonPressMask | ButtonReleaseMask |
                                           EnterWindowMask | LeaveWindowMask);
            u64 WindowAttributeMask = CWBitGravity | CWBackPixel | CWColormap | CWEventMask;
            
            Result.WindowHandle = XCreateWindow(Result.DisplayHandle, RootWindow,
                                                0, 0,
                                                Buffer->Width, Buffer->Height,
                                                0,
                                                WindowVisualInfo.depth, InputOutput,
                                                WindowVisualInfo.visual, WindowAttributeMask, &WindowAttributes);
            if(Result.WindowHandle)
            {
                XRet = XStoreName(Result.DisplayHandle, Result.WindowHandle, "Handmade Window");
                
                // NOTE(luca): If we set the MaxWidth and MaxHeigth to the same values as MinWidth and MinHeight there is a bug on dwm where it won't update the window decorations when trying to remove them.
                // In the future we will allow resizing to any size so this does not matter that much.
#if 0                    
                LinuxSetSizeHint(Result.DisplayHandle, Result.WindowHandle, 0, 0, 0, 0);
#endif
                
                // NOTE(luca): Tiling window managers should treat windows with the WM_TRANSIENT_FOR property as pop-up windows.  This way we ensure that we will be a floating window.  This works on my setup (dwm). 
                XRet = XSetTransientForHint(Result.DisplayHandle, Result.WindowHandle, RootWindow);
                
                XClassHint ClassHint = {};
                ClassHint.res_name = "Handmade Window";
                ClassHint.res_class = "Handmade Window";
                XSetClassHint(Result.DisplayHandle, Result.WindowHandle, &ClassHint);
                
                XSetLocaleModifiers("");
                
                XIM InputMethod = XOpenIM(Result.DisplayHandle, 0, 0, 0);
                if(!InputMethod){
                    XSetLocaleModifiers("@im=none");
                    InputMethod = XOpenIM(Result.DisplayHandle, 0, 0, 0);
                }
                Result.InputContext = XCreateIC(InputMethod,
                                                XNInputStyle, XIMPreeditNothing | XIMStatusNothing,
                                                XNClientWindow, Result.WindowHandle,
                                                XNFocusWindow,  Result.WindowHandle,
                                                NULL);
                XSetICFocus(Result.InputContext);
                
                s32 BitsPerPixel = Buffer->BytesPerPixel*8;
                Result.WindowImage = XCreateImage(Result.DisplayHandle, WindowVisualInfo.visual, WindowVisualInfo.depth, ZPixmap, 0, (char *)Buffer->Pixels, Buffer->Width, Buffer->Height, BitsPerPixel, 0);
                Result.DefaultGC = DefaultGC(Result.DisplayHandle, Screen);
                
                XRet = XMapWindow(Result.DisplayHandle, Result.WindowHandle);
                XRet = XFlush(Result.DisplayHandle);
                
                Result.WM_DELETE_WINDOW = XInternAtom(Result.DisplayHandle, "WM_DELETE_WINDOW", False);
                XRet = XSetWMProtocols(Result.DisplayHandle, Result.WindowHandle, 
                                       &Result.WM_DELETE_WINDOW, 1);
                Assert(XRet);
                Result.Initialized = true;
                
            }
        }
    }
    
    return Result;
}

internal void 
LinuxProcessPendingMessages(linux_x11_context *Linux, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running)
{
	if(Linux->Initialized)
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
LinuxUpdateImage(linux_x11_context *Linux, app_offscreen_buffer *Buffer)
{
	if(Linux->Initialized)
	{
        XPutImage(Linux->DisplayHandle,
                  Linux->WindowHandle, 
                  Linux->DefaultGC, 
                  Linux->WindowImage, 0, 0, 0, 0, 
                  Buffer->Width, 
                  Buffer->Height);
	}
}
