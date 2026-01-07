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
    // NOTE(luca): When looking at the texture in renderdoc I noticed it was flipped.
    v4 TexColor = texture(Texture, v2(TexCoord.x, 1.0 - TexCoord.y));
    FragColor = TexColor;
#else
    FragColor = v4(Color, 1.0f);
#endif
}