typedef struct win32_context win32_context;
struct win32_context
{
    HWND Window;
    HDC OwnDC;
};

global_variable b32 *GlobalRunning;
global_variable b32 GlobalShowCursor;
global_variable s32 GlobalBufferWidth;
global_variable s32 GlobalBufferHeight;

internal LRESULT CALLBACK
Win32MainWindowCallback(HWND Window,
                        UINT Message,
                        WPARAM WParam,
                        LPARAM LParam)
{       
    LRESULT Result = 0;
    
    switch(Message)
    {
        case WM_CLOSE:
        {
            // TODO(casey): Handle this with a message to the user?
            *GlobalRunning = false;
        } break;
        
        case WM_SIZE:
        {
            GlobalBufferWidth = LOWORD(LParam);
            GlobalBufferHeight = HIWORD(LParam);
        } break;
        
        case WM_SETCURSOR:
        {
            if(GlobalShowCursor)
            {
                Result = DefWindowProcA(Window, Message, WParam, LParam);
            }
            else
            {
                SetCursor(0);
            }
        } break;
        
        case WM_DESTROY:
        {
            // TODO(casey): Handle this as an error - recreate window?
            *GlobalRunning = false;
        } break;
        
        case WM_SYSKEYDOWN:
        case WM_SYSKEYUP:
        case WM_KEYDOWN:
        case WM_KEYUP:
        {
            TrapMsg("Keyboard input came in through a non-dispatch message!");
        } break;
        
        default:
        {
            Result = DefWindowProcA(Window, Message, WParam, LParam);
        } break;
    }
    
    return Result;
}

internal P_context 
P_ContextInit(arena *Arena, app_offscreen_buffer *Buffer, b32 *Running)
{
    P_context Result = {};
    
    win32_context *Context = PushStruct(Arena, win32_context);
    
    GlobalShowCursor = true;
    GlobalRunning = Running;
    
    HINSTANCE Instance = GetModuleHandle(0);
    
    WNDCLASSA WindowClass = {};
    WindowClass.style = CS_HREDRAW|CS_VREDRAW;
    WindowClass.lpfnWndProc = Win32MainWindowCallback;
    WindowClass.hInstance = Instance;
    WindowClass.hCursor = LoadCursor(0, IDC_ARROW);
    //    WindowClass.hIcon;
    WindowClass.lpszClassName = "HandmadeHeroWindowClass";
    if(RegisterClassA(&WindowClass))
    {
        HWND Window = CreateWindowExA(0, // WS_EX_TOPMOST|WS_EX_LAYERED,
                                      WindowClass.lpszClassName,
                                      "Handmade Hero",
                                      WS_OVERLAPPEDWINDOW|WS_VISIBLE,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      Buffer->Width,
                                      Buffer->Height,
                                      0,
                                      0,
                                      Instance,
                                      0);
        if(Window)
        {
            HGLRC GLContext;
            
            HDC OwnDC = GetDC(Window);
            
            int Win32RefreshRate = GetDeviceCaps(OwnDC, VREFRESH);
            // TODO(luca): Use this one.
            
            PIXELFORMATDESCRIPTOR PixelFormat = {};
            PixelFormat.nSize = sizeof(PixelFormat);
            PixelFormat.nVersion = 1;
            PixelFormat.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            PixelFormat.iPixelType = PFD_TYPE_RGBA;
            PixelFormat.cColorBits = 32;
            PixelFormat.cDepthBits = 24;
            PixelFormat.cStencilBits = 8;
            PixelFormat.iLayerType = PFD_MAIN_PLANE;
            
            int ChosenFormat = ChoosePixelFormat(OwnDC, &PixelFormat);
            SetPixelFormat(OwnDC, ChosenFormat, &PixelFormat);
            
            GLContext = wglCreateContext(OwnDC);
            wglMakeCurrent(OwnDC, GLContext);
            
            Context->OwnDC = OwnDC;
            Context->Window = Window;
            
            Result = (umm)Context;
            
        }
    }
    
    return Result;
}

internal void      
P_UpdateImage(P_context Context, app_offscreen_buffer *Buffer)
{
    win32_context *Win32 = (win32_context *)Context; 
    if(Win32)
    {
        SwapBuffers(Win32->OwnDC);
    }
}

internal void      
P_ProcessMessages(P_context Context, app_input *Input, app_offscreen_buffer *Buffer, b32 *Running)
{
    win32_context *Win32Context = (win32_context *)Context;
    
    if(Win32Context)
    {    
        MSG Message;
        while(PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
        {
            switch(Message.message)
            {
                case WM_QUIT:
                {
                    *GlobalRunning = false;
                } break;
                
                case WM_SYSKEYDOWN:
                case WM_SYSKEYUP:
                case WM_KEYDOWN:
                case WM_KEYUP:
                {
                    u32 VKCode = (u32)Message.wParam;
                    
                    b32 WasDown = ((Message.lParam & (1 << 30)) != 0);
                    b32 IsDown = ((Message.lParam & (1 << 31)) == 0);
                    
                    if(IsDown)
                    {
                        b32 Shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
                        
                        b32 Alt = (Message.lParam & (1 << 29));
                        
                        if((VKCode == VK_F4) && Alt)
                        {
                            *GlobalRunning = false;
                        }
                        
                        app_text_button *Button = &Input->Text.Buffer[Input->Text.Count];
                        *Button = {};
                        Input->Text.Count += 1;
                        
                        if(Alt)   Button->Modifiers |= PlatformKeyModifier_Alt;
                        if(Shift) Button->Modifiers |= PlatformKeyModifier_Shift;
                        
                        if((VKCode >= 'A') && (VKCode <= 'Z'))
                        {
                            Button->Codepoint = ((VKCode - 'A') + 'a');
                        }
                        else
                        {
                            Button->IsSymbol = true;
                            if(0) {}
                            else if(VKCode == VK_UP) Button->Symbol = PlatformKey_Up;
                            else if(VKCode == VK_DOWN) Button->Symbol = PlatformKey_Down;
                            else if(VKCode == VK_LEFT) Button->Symbol = PlatformKey_Left;
                            else if(VKCode == VK_RIGHT) Button->Symbol = PlatformKey_Right;
                            
                        }
                    }
                } break;
                
                default:
                {
                    TranslateMessage(&Message);
                    DispatchMessageA(&Message);
                } break;
            }
        }
        
        Buffer->Width = GlobalBufferWidth;
        Buffer->Height = GlobalBufferHeight;
        
        // Mouse
        {        
            POINT MouseP;
            GetCursorPos(&MouseP);
            ScreenToClient(Win32Context->Window, &MouseP);
            
            if(MouseP.x >= 0 && MouseP.x < Buffer->Width &&
               MouseP.y >= 0 && MouseP.y < Buffer->Height)
            {                    
                Input->MouseX = MouseP.x;
                Input->MouseY = MouseP.y;
            }
            
            // TODO(luca): Support mousewheel
            Input->MouseZ = 0; 
            
            ProcessKeyPress(&Input->Buttons[0], GetKeyState(VK_LBUTTON) & (1 << 15));
            ProcessKeyPress(&Input->Buttons[1], GetKeyState(VK_MBUTTON) & (1 << 15));
            ProcessKeyPress(&Input->Buttons[2], GetKeyState(VK_RBUTTON) & (1 << 15));
            ProcessKeyPress(&Input->Buttons[3], GetKeyState(VK_XBUTTON1) & (1 << 15));
            ProcessKeyPress(&Input->Buttons[4], GetKeyState(VK_XBUTTON2) & (1 << 15));
        }
        
    }
}

internal void
P_LoadAppCode(arena *Arena, app_code *Code, app_state *App, s64 *LastWriteTime)
{
    HMODULE Library = (HMODULE)Code->LibraryHandle;
    
    char *LockFileName = PathFromExe(Arena, App, S8("lock.tmp"));
    char *TempDLLPath = PathFromExe(Arena, App, S8("ex_app_temp.dll"));
    
    WIN32_FILE_ATTRIBUTE_DATA Data;
    
    WIN32_FILE_ATTRIBUTE_DATA Ignored;
    if(!GetFileAttributesEx(LockFileName, GetFileExInfoStandard, &Ignored))
    {
        
        s64 WriteTime = *LastWriteTime;
        if(GetFileAttributesEx(Code->LibraryPath, GetFileExInfoStandard, &Data))
        {
            WriteTime = (s64)((((u64)Data.ftLastWriteTime.dwHighDateTime & 0xFFFF) << 32) | 
                              (((u64)Data.ftLastWriteTime.dwLowDateTime & 0xFFFF) << 0));
        }
        
        if(*LastWriteTime != WriteTime)
        {
            *LastWriteTime = WriteTime;
            
            if(Library)
            {
                FreeLibrary(Library);
                Code->Loaded = false;
                Code->LibraryHandle = 0;
            }
            
            b32 Result = CopyFile(Code->LibraryPath, TempDLLPath, FALSE);
            if(!Result)
            {
                DebugBreak;
                Win32LogIfError();
            }
            
            Library = LoadLibraryA(TempDLLPath);
            if(Library)
            {
                Code->UpdateAndRender = (update_and_render *)GetProcAddress(Library, "UpdateAndRender");
                if(Code->UpdateAndRender)
                {
                    Code->Loaded = true;
                    App->Reloaded = true;
                    Code->LibraryHandle = (umm)Library;
                    Log("\nLibrary reloaded.\n");
                }
                else
                {
                    Code->Loaded = false;
                    ErrorLog("Could not find UpdateAndRender.\n");
                }
            }
            else
            {
                Code->Loaded = false;
                ErrorLog("Could not open library.\n");
            }
        }
    }
    
    if(!Code->Loaded)
    {
        Code->UpdateAndRender = UpdateAndRenderStub;
    }
}
