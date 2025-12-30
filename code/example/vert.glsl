#version 330 core

layout (location = 0) in vec3 pos;

uniform vec3 color;
uniform vec2 angle;

out vec3 Color;

float deg2rad(float degrees)
{
    float Result = degrees*3.14159265359f/180.0f;
    return Result;
}

vec3 rotate(vec3 Pos, float Angle)
{
    vec3 Result;
    
    float c = cos(Angle);
    float s = sin(Angle);
    Result.x = Pos.x*c - Pos.z*s;
    Result.z = Pos.x*s + Pos.z*c;
    Result.y = Pos.y;
    
    return Result;
}

void main()
{
    Color = color;
    
#if 1    
    float x, y, z;;
    
    vec3 inc = rotate(pos.yxz, angle.y).yxz;
    vec3 rot = rotate(inc, angle.x);
    x = rot.x;
    y = rot.y;
    z = rot.z;
    
    float depth = z + 5.0;
    gl_Position = vec4(x/depth, y/depth, z, 1.0);
#else
    gl_Position = vec4(pos, 1.0f);
#endif
    
}