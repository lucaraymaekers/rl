#ifndef BASE_TYPES_H
#define BASE_TYPES_H

#include <stdint.h>
#include <stddef.h>
#if OS_LINUX
# include <sys/types.h>
#elif OS_WINDOWS
# include <Windows.h>
# define ssize_t SSIZE_T
#endif

#define internal static 
#define local_persist static 
#define global_variable static
#define thread_static __thread

#define Pi32 3.14159265359f

#define Kilobytes(Value) ((Value)*1024LL)
#define Megabytes(Value) (Kilobytes(Value)*1024LL)
#define Gigabytes(Value) (Megabytes(Value)*1024LL)
#define Terabytes(Value) (Gigabytes(Value)*1024LL)

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef s32 b32;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef size_t umm;
typedef ssize_t smm;
typedef s32 rune; // utf8 codepoint

typedef float f32;
typedef double f64;

#define U8Max 0xff
#define U16Max 0xffff
#define S32Min ((s32)0x80000000)
#define S32Max ((s32)0x7fffffff)
#define U32Min 0
#define U32Max ((u32)-1)
#define U64Max ((u64)-1)

#define false 0
#define true  1

typedef struct range_s64 range_s64;
struct range_s64
{
    s64 Min;
    s64 Max;
};

typedef struct str8 str8;
struct str8
{
    umm Size;
    u8 *Data;
};
raddbg_type_view(str8, no_addr(array((char *)Data, Size)));
#define S8Lit(String) (str8){.Data = (u8 *)(String), .Size = (sizeof((String)) - 1)}

#endif //BASE_TYPES_H
