#include "base/base.h"

#if OS_WINDOWS
# define SLASH "\\"
#elif OS_LINUX
# define SLASH "/"
#endif

#define CLING_CODE_PATH   ".." SLASH "code" SLASH
#define CLING_SOURCE_PATH CLING_CODE_PATH "cling" SLASH "cling.c"
#include "cling.h"

#define BUILD ".." SLASH "build" SLASH

//~ Globals
global_variable arena *GlobalClingArena;

//~ Functions
void Build(arena *Arena, char *Env[], str8 Source, str8 ExtraFlags)
{
    
    str8 Output = PushS8(Arena, KB(2)); 
    
    printf(S8Fmt "\n", S8Arg(OS_GetFileName(Source)));
    
    str8_list BuildCommandList = CommonBuildCommand(Arena, false, true ,true);
    
    Str8ListAppendMultiple(&BuildCommandList, ExtraFlags, Source);
    str8 BuildCommand = Str8ListJoin(BuildCommandList, Output, ' ');
    
    os_command_result CommandResult = OS_RunCommand(BuildCommand, Env, true);
    smm BytesRead = LinuxErrorWrapperRead(CommandResult.Stderr, Output.Data, CommandResult.StderrBytesToRead);
    if(BytesRead)
    {
        printf(S8Fmt, (int)BytesRead, (char *)Output.Data);
    }
}

//- Entrypoint 

int 
main(int ArgsCount, char *Args[], char *Env[])
{
    GlobalClingArena = ArenaAlloc();
    
    arena *Arena = GlobalClingArena;
    
    str8 Temp = PushS8(Arena, KB(2));
    str8 Output = PushS8(Arena, KB(2));
    
    OS_RebuildSelf(Arena, ArgsCount, Args, Env);
    
    str8 CodePath = OS_PathFromExe(Temp.Data, Args[0], S8(CLING_CODE_PATH));
    OS_ChangeDirectory((char *)CodePath.Data);
    
    b32 Example = true;
    b32 App = true;
    b32 Sort = false;
    b32 Hash = false;
    
    b32 HasTarget = ((Example && (App || Sort)) || 
                     Hash);
    
    if(HasTarget)
    {
        printf("[debug mode]\n"
               "[clang compile]\n");
    }
    
    if(Example)
    {    
        if(App)
        {    
            str8 File = OS_ReadEntireFileIntoMemory(BUILD "rl_libs.o");
            if(!File.Size)
            {                
                Build(Arena, Env, S8("./lib/rl_libs.h"),
                      S8("-fPIC -D_GNU_SOURCE=1 "
                         "-c -x c++ "
                         "-I . "
                         "-Wno-unused-command-line-argument "
                         "-o " BUILD "rl_libs.o"));
            }
            
            Build(Arena, Env, S8("./example/ex_app.cpp"),
                  S8("-o " BUILD "ex_app.so "
                     "-DRL_FAST_COMPILE=1 "
                     "-I . "
                     "-fPIC -shared "
                     BUILD "rl_libs.o"));
            
            Build(Arena, Env, S8("./example/ex_platform.cpp"),
                  S8("-o " BUILD "ex_platform "
                     "-I . "
                     "-lX11 -lGL -lGLX "));
        }
        
        if(Sort)
        {
            Build(Arena, Env, S8("./example/sort/ex_app.cpp"),
                  S8("-o " BUILD "ex_app.so "
                     "-DRL_FAST_COMPILE=1 "
                     "-I . "
                     "-fPIC -shared "
                     BUILD "rl_libs.o"));
            
            Build(Arena, Env, S8("./example/sort/ex_platform.cpp"),
                  S8("-o " BUILD "ex_platform "
                     "-I . "
                     "-lX11 -lGLX -lGL"));
        }
    }
    
    if(Hash)
    {
        Build(Arena, Env, S8("./hash/hash.c"),
              S8("-o " BUILD "hash "
                 "-I ./"));
    }
    
    return 0;
}
