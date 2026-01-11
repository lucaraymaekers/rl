#version 330 core

#define v2 vec2
#define v3 vec3
#define v4 vec4 
#define f32 float

in v3 pos;
in v3 color;
in v2 buttonMin;
in v2 buttonMax;

flat out v3 Color;
out v2 LocalPos;
flat out v2 ButtonMin;
flat out v2 ButtonMax;

void main()
{
    Color = color;
    ButtonMin = buttonMin;
    ButtonMax = buttonMax;
    
    LocalPos = pos.xy;
    
    gl_Position = v4(pos, 1.0f);
}