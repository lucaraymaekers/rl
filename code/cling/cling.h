//~ Libraries
// Standard library (TODO(luca): get rid of it) 
#include <stdio.h>
#include <string.h>

#if !defined(CLING_SOURCE_PATH)
# error "You must define CLING_SOURCE_PATH"
#endif

#if !defined(CLING_CODE_PATH)
# error "You must define CLING_CODE_PATH"
#endif

#define BASE_NO_ENTRYPOINT 1
#include "base/base.h"
#include "base/base.c"
//~ Types

struct str8_list
{
    str8 *Strings;
    umm Count;
    umm Capacity;
};
typedef struct str8_list str8_list;

//~ Strings
internal b32 
IsWhiteSpace(u8 Char)
{
    b32 Result = (Char == ' ' || Char == '\t' || Char == '\n' || Char == '\r');
    return Result;
}

internal void 
Str8ListAppend(str8_list *List, str8 String)
{
    Assert(List->Count < List->Capacity);
    List->Strings[List->Count++] = String;
}

#define Str8ListAppendMultiple(List, ...) \
do \
{ \
str8 Strings[] = {__VA_ARGS__}; \
_Str8ListAppendMultiple(List, ArrayCount(Strings), Strings); \
} while(0);
void _Str8ListAppendMultiple(str8_list *List, umm Count, str8 *Strings)
{
    for EachIndex(At, Count)
    {
        Str8ListAppend(List, Strings[At]);
    }
}

internal str8 
Str8ListJoin(str8_list List, str8 Buffer, u8 Char)
{
    str8 Result = {};
    
    umm BufferIndex = 0;
    for EachIndex(At, List.Count)
    {
        str8 *StringAt = List.Strings + At;
        
        Assert(BufferIndex + StringAt->Size < Buffer.Size);
        MemoryCopy(Buffer.Data + BufferIndex, StringAt->Data, StringAt->Size);
        BufferIndex += StringAt->Size;
        if(Char)
        {
            Buffer.Data[BufferIndex] = Char;
            BufferIndex += 1;
        }
    }
    
    Result.Data = Buffer.Data;
    Result.Size = BufferIndex;
    
    return Result;
}

//~ Helpers
internal str8_list 
CommonBuildCommand(arena *Arena, b32 GCC, b32 Clang, b32 Debug)
{
    str8_list BuildCommand = {};
    
    // Exclusive arguments
    if(GCC) Clang = false;
    b32 Release = !Debug;
    
    BuildCommand.Capacity = 256;
    BuildCommand.Strings = PushArray(Arena, str8, BuildCommand.Capacity);
    
    str8 CommonCompilerFlags = S8("-fsanitize-trap -nostdinc++ -D_GNU_SOURCE=1");
    str8 CommonWarningFlags = S8("-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-sign-conversion -Wno-sign-compare -Wno-double-promotion -Wno-unused-but-set-variable -Wno-unused-variable -Wno-write-strings -Wno-missing-field-initializers -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function");
    
    str8 LinuxLinkerFlags = S8("-lpthread -lm");
    
    str8 Compiler = {};
    str8 Mode = {};
    
    if(0) {}
    else if(Release)
    {
        Mode = S8("release");
    }
    else if(Debug)
    {
        Mode = S8("debug");
    }
    
    if(0) {}
    else if(Clang)
    {
        Compiler = S8("clang");
        Str8ListAppend(&BuildCommand, Compiler);
        if(Debug)
        {
            Str8ListAppend(&BuildCommand, S8("-g -ggdb -g3"));
        }
        else if(Release)
        {
            Str8ListAppend(&BuildCommand, S8("-O3"));
        }
        Str8ListAppend(&BuildCommand, CommonCompilerFlags);
        Str8ListAppend(&BuildCommand, S8("-fdiagnostics-absolute-paths -ftime-trace"));
        Str8ListAppend(&BuildCommand, CommonWarningFlags);
        Str8ListAppend(&BuildCommand, S8("-Wno-null-dereference -Wno-missing-braces -Wno-vla-cxx-extension -Wno-writable-strings -Wno-missing-designated-field-initializers -Wno-address-of-temporary -Wno-int-to-void-pointer-cast"));
    }
    else if(GCC)
    {
        Compiler = S8("g++");
        Str8ListAppend(&BuildCommand, Compiler);
        
        if(Debug)
        {
            Str8ListAppend(&BuildCommand, S8("-g -ggdb -g3"));
        }
        else if(Release)
        {
            Str8ListAppend(&BuildCommand, S8("-O3"));
        }
        Str8ListAppend(&BuildCommand, CommonCompilerFlags);
        Str8ListAppend(&BuildCommand, CommonWarningFlags);
        Str8ListAppend(&BuildCommand, S8("-Wno-cast-function-type -Wno-missing-field-initializers -Wno-int-to-pointer-cast"));
    }
    
#if OS_LINUX    
    Str8ListAppend(&BuildCommand, LinuxLinkerFlags);
#elif OS_WINDOWS    
    ZeroMemory(&BuildCommand, sizeof(BuildCommand));
    BuildCommand.Strings = (str8 *)StringsBuffer.Data;
    BuildCommand.Capacity = StringsBuffer.Size/sizeof(str8);
    
    Str8ListAppend(&BuildCommand, S8("cl"));
    Str8ListAppend(&BuildCommand, S8("-MTd -Gm- -nologo -GR- -EHa- -Oi -FC -Z7"));
    Str8ListAppend(&BuildCommand, S8("-WX -W4 -wd4459 -wd4456 -wd4201 -wd4100 -wd4101 -wd4189 -wd4505 -wd4996 -wd4389 -wd4244"));
    Str8ListAppend(&BuildCommand, S8("-I..\\.."));
#endif
    
    return BuildCommand;
}

#define CLING_OS_IMPLEMENTATION 1
#include "cling_os.c"