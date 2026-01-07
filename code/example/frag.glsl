#version 330 core

#define v2 vec2
#define v3 vec3
#define v4 vec4

out v4 FragColor;

in v3 Color;
in v2 TexCoord;

uniform sampler2D Texture;

void main()
{
#if 1
    v4 TexColor = texture(Texture, TexCoord);
    FragColor = TexColor;
#else
    FragColor = v4(Color, 1.0f);
#endif
}