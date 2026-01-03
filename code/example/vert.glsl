#version 330 core

#define v2 vec2
#define v3 vec3
#define v4 vec4 
#define f32 float

layout (location = 0) in v3 pos;

uniform v3 color;
uniform v2 angle;

out v3 Color;

f32 deg2rad(f32 degrees)
{
    f32 Result = degrees*3.14159265359f/180.0f;
    return Result;
}

v3 rotate(v3 Pos, f32 Angle)
{
    v3 Result;
    
    f32 c = cos(Angle);
    f32 s = sin(Angle);
    Result.x = Pos.x*c - Pos.z*s;
    Result.z = Pos.x*s + Pos.z*c;
    Result.y = Pos.y;
    
    return Result;
}

void main()
{
    Color = color;
    
#if 1   
    f32 x, y, z;;
    
    v3 inc = rotate(pos.yxz, angle.y).yxz;
    v3 rot = rotate(inc, angle.x);
    x = rot.x;
    y = rot.y;
    z = rot.z;
    
    f32 depth = z + 3.0;
    gl_Position = v4(x/depth, y/depth, 0.0, 1.0);
#else
    gl_Position = v4(pos, 1.0f);
#endif
    
}