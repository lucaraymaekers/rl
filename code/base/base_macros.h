#ifndef BASE_MACROS_H
#define BASE_MACROS_H

// Detect compiler
#if __clang__

# define COMPILER_CLANG 1
#elif _MSC_VER
# define COMPILER_MSVC 1
#elif __GNUC__
# define COMPILER_GNU 1
#endif

// Zero undefined
#if !defined(COMPILER_MSVC)
# define COMPILER_MSVC 0
#endif
#if !defined(COMPILER_LLVM)
# define COMPILER_LLVM 0
#endif
#if !defined(COMPILER_GNU)
# define COMPILER_GNU 0
#endif

// Push/Pop warnings
#if COMPILER_GNU
# define PUSH_WARNINGS \
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic ignored \"-Wall\"") \
_Pragma("GCC diagnostic ignored \"-Wextra\"") \
_Pragma("GCC diagnostic ignored \"-Wconversion\"") \
_Pragma("GCC diagnostic ignored \"-Wfloat-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-compare\"") \
_Pragma("GCC diagnostic ignored \"-Wdouble-promotion\"") \
_Pragma("GCC diagnostic ignored \"-Wimplicit-fallthrough\"")
# define POP_WARNINGS _Pragma("GCC diagnostic pop")

#elif COMPILER_CLANG
# define PUSH_WARNINGS \
_Pragma("clang diagnostic push") \
_Pragma("clang diagnostic ignored \"-Weverything\"")
# define POP_WARNINGS _Pragma("clang diagnostic pop")

#else
# error "No compatible compiler found"
#endif

#if __cplusplus
# define C_LINKAGE extern "C"
#else
# define C_LINKAGE
#endif

#define ERROR_FMT "%s(%d): ERROR: "
#define ERROR_ARG __FILE__, __LINE__

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))


//~ Globals
static int GlobalDebuggerIsAttached;

#if RL_INTERNAL
# define Assert(Expression) \
do {\
if(!(Expression))\
{\
if(GlobalDebuggerIsAttached) {__asm__ volatile("int3"); }\
else                         { ErrorLog("Hit assertion\n"); }\
}\
} while(0)
#else
# define Assert(Expression)
#endif

#define DebugBreak     do { Assert(0); } while(0)
#define DebugBreakOnce { local_persist b32 X = false; Assert(X); X = true; }
#define NullExpression do { int X = 0; } while(0)

#define Minimum(A, B) (((A) < (B)) ? (A) : (B))
#define Maximum(A, B) (((A) > (B)) ? (A) : (B))
#define Swap(A, B)    do { typeof(A) temp = (typeof(A))A; A = B; B = temp; } while(0)

#define EachIndex(Index, Count) umm Index = 0; Index < (Count); Index += 1

#if OS_WINDOWS
# include <windows.h>
# define RADDBG_MARKUP_IMPLEMENTATION
#else
# define RADDBG_MARKUP_STUBS
#endif
#include "lib/raddbg_markup.h"

#endif // BASE_MACROS_H