/* date = January 11th 2026 11:03 pm */

#ifndef EX_MATH_H
#define EX_MATH_H
internal inline v2 
V2AddV2(v2 A, v2 B)
{
    v2 Result = {};
    Result.X = A.X + B.X;
    Result.Y = A.Y + B.Y;
    return Result;
}

internal inline v2 
V2AddF32(v2 A, f32 B)
{
    v2 Result = {};
    Result.X = A.X + B;
    Result.Y = A.Y + B;
    return Result;
}

internal inline v2 
V2MulF32(v2 A, f32 B)
{
    v2 Result = {};
    Result.X = A.X * B;
    Result.Y = A.Y * B;
    return Result;
}

internal inline v2
V2SubV2(v2 A, v2 B)
{
    v2 Result = {};
    Result.X = A.X - B.X;
    Result.Y = A.Y - B.Y;
    return Result;
}

internal inline v2
V2S32(s32 X, s32 Y)
{
    v2 Result = {};
    Result.X = (f32)X;
    Result.Y = (f32)Y;
    return Result;
}

internal inline v2
V2MulV2(v2 A, v2 B)
{
    v2 Result = {};
    Result.X = A.X * B.X;
    Result.Y = A.Y * B.Y;
    return Result;
}

internal inline v2 
V2(f32 A, f32 B) 
{
    v2 Result = {};
    Result.X = A;
    Result.Y = B;
    return Result;
}

internal inline v3 
V3(f32 X, f32 Y, f32 Z) 
{
    v3 Result = {};
    Result.X = X;
    Result.Y = Y;
    Result.Z = Z;
    return Result;
}

internal inline b32
InBounds(v2 A, v2 Min, v2 Max)
{
    b32 Result = !!((A.X >= Min.X && A.X < Max.X) &&
                    (A.Y >= Min.Y && A.Y < Max.Y));
    return Result;
}

internal inline void
SetProvokingV3(v3 Quad[6], v3 Vertex)
{
    Quad[2] = Vertex;
    Quad[5] = Vertex;
}

internal inline void
SetProvokingV2(v2 Quad[6], v2 Vertex)
{
    Quad[2] = Vertex;
    Quad[5] = Vertex;
}

#endif //EX_MATH_H
