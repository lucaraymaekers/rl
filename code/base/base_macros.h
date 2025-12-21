#ifndef BASE_MACROS_H
#define BASE_MACROS_H

// detect OS
#if __linux__
# define OS_LINUX 1
#elif _WIN32
# define OS_WINDOWS 1
#endif

// Detect compiler
#if __clang__
# define COMPILER_CLANG 1
#elif _MSC_VER
# define COMPILER_MSVC 1
#elif __GNUC__
# define COMPILER_GNU 1
#endif

// Zero undefined
#ifndef OS_LINUX
# define OS_LINUX 0
#endif
#ifndef OS_WINDOWS
# define OS_WINDOWS 0
#endif
#ifndef COMPILER_MSVC
# define COMPILER_MSVC 0
#endif
#ifndef COMPILER_LLVM
# define COMPILER_LLVM 0
#endif
#ifndef COMPILER_GNU
# define COMPILER_GNU 0
#endif

#if OS_WINDOWS
# include <windows.h>
# define RADDBG_MARKUP_IMPLEMENTATION
#else
# define RADDBG_MARKUP_STUBS
#endif
#include "lib/raddbg_markup.h"

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

#elif COMPILER_MSVC
# define PUSH_WARNINGS \
__pragma(warning(push)) \
__pragma(warning(disable: 4267 4996)) // Add specific warning numbers to disable as needed

# define POP_WARNINGS __pragma(warning(pop))

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

#if OS_LINUX
# define DebugBreak do { if(GlobalDebuggerIsAttached) __asm__ volatile("int3"); } while(0)
#elif OS_WINDOWS
# define DebugBreak do { if(GlobalDebuggerIsAttached) DebugBreak(); } while(0)
#endif

#if RL_INTERNAL
# define AssertMsg(Expression, Format, ...) \
do { if(!(Expression)) { ErrorLog(Format, ##__VA_ARGS__); DebugBreak; } } while(0)
# define Assert(Expression) AssertMsg(Expression, "Hit assertion")
#else
# define Assert(Expression)
#endif

#define DebugBreakOnce { local_persist b32 X = false; Assert(X); X = true; }
#define NullExpression do { int X = 0; } while(0)

#define Minimum(A, B) (((A) < (B)) ? (A) : (B))
#define Maximum(A, B) (((A) > (B)) ? (A) : (B))

#if !defined(__cplusplus)
# define Swap(A, B) do { typeof(A) temp = (typeof(A))A; A = B; B = temp; } while(0)
#else
template <typename type> inline void 
Swap(type& A, type& B) { type T = A; A = B; B = T; }
#endif

#define EachIndex(Index, Count) (umm Index = 0; Index < (Count); Index += 1)

#define MemoryCopy memcpy
#define MemorySet  memset
#define MemoryMove memmove

#endif // BASE_MACROS_H