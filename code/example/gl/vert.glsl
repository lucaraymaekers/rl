#version 330 core

in vec2 InPos;
in vec3 InColor;
in vec2 InTexCoord;

out vec3 Color;
out vec2 TexCoord;

void main()
{
    Color = InColor;
    TexCoord = InTexCoord;
    
    gl_Position = vec4(InPos, 0.0f, 1.0f);
}