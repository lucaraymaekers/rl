#version 330 core

#define v2 vec2
#define v3 vec3
#define v4 vec4 
#define f32 float

in v3 Color;
in v2 LocalPos;
in v2 ButtonMin;
in v2 ButtonMax;

out v4 FragColor;

// NOTE(luca): Center is at 0.0 between -1.0 and 1.0
f32 DistanceFromPoint(v2 Center, f32 Radius)
{
    v2 Q = abs(Center) - v2(1.0f) + Radius;
    return length(max(Q, 0.0)) - Radius;
}

void main()
{
    v2 Min = (ButtonMin*2.0f - 1.0f)*v2(1.0f, -1.0f);
    v2 Max = (ButtonMax*2.0f - 1.0f)*v2(1.0f, -1.0f);
    
    v2 Size = Max - Min;
    v2 Pos = (LocalPos - Min);
    
    v2 Bilateral = 2.0f*(Pos/Size) - 1.0f;
    
    f32 Distance = DistanceFromPoint(Bilateral, 0.2f);
    f32 Alpha = 1.0 - smoothstep(0.0, 0.01, Distance);
    
    FragColor = v4(Color, Alpha);
    
}