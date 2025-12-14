#include <dlfcn.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>
#include <X11/extensions/Xrandr.h>
#include <X11/cursorfont.h>

#include "base/base.h"
#include "cu.h"
#include "platform.h"

PUSH_WARNINGS;
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"
POP_WARNINGS;

global_variable b32 GlobalRunning = true;

CU_UPDATE_AND_RENDER(UpdateAndRenderStub)
{
    
}

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
LinuxProcessKeyPress(game_button_state *ButtonState, b32 IsDown)
{
    if(ButtonState->EndedDown != IsDown)
    {
        ButtonState->EndedDown = IsDown;
        ButtonState->HalfTransitionCount++;
    }
}

internal void 
LinuxProcessPendingMessages(Display *DisplayHandle, Window WindowHandle, s32 Width, s32 Height,
                            XIC InputContext, Atom WM_DELETE_WINDOW, game_input *Input)
{
    XEvent WindowEvent = {};
    while(XPending(DisplayHandle) > 0)
    {
        XNextEvent(DisplayHandle, &WindowEvent);
        b32 FilteredEvent = XFilterEvent(&WindowEvent, 0);
        if(FilteredEvent)
        {
            Assert(WindowEvent.type == KeyPress || WindowEvent.type == KeyRelease);
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
                    
                    s32 BytesLookepdUp = Xutf8LookupString(InputContext, &WindowEvent.xkey, 
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
                                game_text_button *TextButton = &Input->Text.Buffer[Input->Text.Count++];
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
                    GlobalRunning = false;
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
                    LinuxProcessKeyPress(&Input->MouseButtons[PlatformMouseButton_Left], IsDown);
                }
                else if(Button == Button2)
                {
                    LinuxProcessKeyPress(&Input->MouseButtons[PlatformMouseButton_Middle], IsDown);
                }
                else if(Button == Button3)
                {
                    LinuxProcessKeyPress(&Input->MouseButtons[PlatformMouseButton_Right], IsDown);
                }
                else if(Button == Button4)
                {
                    LinuxProcessKeyPress(&Input->MouseButtons[PlatformMouseButton_ScrollUp], IsDown);
                }
                else if(Button == Button5)
                {
                    LinuxProcessKeyPress(&Input->MouseButtons[PlatformMouseButton_ScrollDown], IsDown);
                }
            } break;
            
            case DestroyNotify:
            {
                XDestroyWindowEvent *Event = (XDestroyWindowEvent *)&WindowEvent;
                if(Event->window == WindowHandle)
                {
                    GlobalRunning = false;
                }
            } break;
            
            case ClientMessage:
            {
                XClientMessageEvent *Event = (XClientMessageEvent *)&WindowEvent;
                if((Atom)Event->data.l[0] == WM_DELETE_WINDOW)
                {
                    XDestroyWindow(DisplayHandle, WindowHandle);
                    GlobalRunning = false;
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
    
    
    
    // TODO(luca): Move this into process pending messages.
    s32 MouseX = 0, MouseY = 0, MouseZ = 0;
    u32 MouseMask = 0;
    u64 Ignored;
    if(XQueryPointer(DisplayHandle, WindowHandle, 
                     &Ignored, &Ignored, (int *)&Ignored, (int *)&Ignored,
                     &MouseX, &MouseY, &MouseMask))
    {
        if(MouseX <= Width && MouseX >= 0 &&
           MouseY <= Height && MouseY >= 0)
        {
            Input->MouseY = MouseY;
            Input->MouseX = MouseX;
        }
    }
}


struct linux_init_x11_result
{
    XImage *WindowImage;
    Display *DisplayHandle;
    Window WindowHandle;
    GC DefaultGC;
    XIC InputContext;
};

linux_init_x11_result LinuxInitX11(u8 *HostPixels, s32 Width, s32 Height)
{
    linux_init_x11_result Result = {};
    
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
                                                Width, Height,
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
                
                Atom WM_DELETE_WINDOW = XInternAtom(Result.DisplayHandle, "WM_DELETE_WINDOW", False);
                if(!XSetWMProtocols(Result.DisplayHandle, Result.WindowHandle, &WM_DELETE_WINDOW, 1))
                {
                    // TODO(luca): Logging
                }
                
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
                
                int BitsPerPixel = 32;
                int BytesPerPixel = BitsPerPixel/8;
                int WindowBufferSize = Width*Height*BytesPerPixel;
                
                Result.WindowImage = XCreateImage(Result.DisplayHandle, WindowVisualInfo.visual, WindowVisualInfo.depth, ZPixmap, 0, (char *)HostPixels, Width, Height, BitsPerPixel, 0);
                Result.DefaultGC = DefaultGC(Result.DisplayHandle, Screen);
                
                XRet = XMapWindow(Result.DisplayHandle, Result.WindowHandle);
                XRet = XFlush(Result.DisplayHandle);
            }
        }
    }
    
    return Result;
}

C_LINKAGE ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        s32 Width = 1920/2;
        s32 Height = 1080/2;
        s32 BytesPerPixel = 4;
        s32 Pitch = Width*BytesPerPixel;
        s32 Size = Width*Height*BytesPerPixel;
        
        CU_Check(cudaSetDevice(0));
        cudaDeviceProp Prop;
        CU_Check(cudaGetDeviceProperties(&Prop, 0));
        
        CU_update_and_render *UpdateAndRender = 0;
        
        arena *CPUArena = GetScratch();
        arena *GPUArena = CU_ArenaAlloc(CPUArena);
        
        u8 *HostPixels = PushArray(CPUArena, u8, Size);
        
        linux_init_x11_result LinuxX11 = LinuxInitX11(HostPixels, Width, Height);
        Atom WM_DELETE_WINDOW = XInternAtom(LinuxX11.DisplayHandle, "WM_DELETE_WINDOW", False);
        
        void *Library = 0;
        
        game_input Input[2] = {};
        game_input *NewInput = &Input[0];
        game_input *OldInput = &Input[1];
        
        GlobalRunning = true;
        while(GlobalRunning)
        {
            BeginScratch(CPUArena);
            BeginScratch(GPUArena);
            
            if(Library)
            {
                dlclose(Library);
            }
            Library = dlopen("./build/sphere.so", RTLD_NOW);
            if(!Library)
            {
                char *Error = dlerror();
                ErrorLog("%s\n", Error);
                UpdateAndRender = UpdateAndRenderStub;
            }
            else
            {
                UpdateAndRender = (CU_update_and_render *)dlsym(Library, "UpdateAndRender");
                if(!UpdateAndRender)
                {
                    ErrorLog("Could not find UpdateAndRender.\n");
                    UpdateAndRender = UpdateAndRenderStub;
                }
            }
            Assert(UpdateAndRender);
            
            
            NewInput->Text.Count = 0;
            for(u32 ButtonIndex = 0; ButtonIndex < PlatformMouseButton_Count; ButtonIndex += 1)
            {
                NewInput->MouseButtons[ButtonIndex].EndedDown = OldInput->MouseButtons[ButtonIndex].EndedDown;
                NewInput->MouseButtons[ButtonIndex].HalfTransitionCount = 0;
            }
            
            LinuxProcessPendingMessages(LinuxX11.DisplayHandle, LinuxX11.WindowHandle, 
                                        Width, Height,
                                        LinuxX11.InputContext, WM_DELETE_WINDOW, NewInput);
            
            OS_PrintFormat("%*s (%d, %d)\n", 
                           NewInput->Text.Count, NewInput->Text.Buffer,
                           NewInput->MouseX, NewInput->MouseY); 
            
            UpdateAndRender(ThreadContext, CPUArena, GPUArena,
                            HostPixels, Width, Height, BytesPerPixel, Pitch);
            
            XPutImage(LinuxX11.DisplayHandle, LinuxX11.WindowHandle, LinuxX11.DefaultGC, LinuxX11.WindowImage, 0, 0, 0, 0, 
                      Width, Height);
            
            game_input *TempInput = NewInput;
            NewInput = OldInput;
            TempInput = NewInput;
            
            EndScratch(CPUArena);
            EndScratch(GPUArena);
        }
    }
    
    return 0;
}