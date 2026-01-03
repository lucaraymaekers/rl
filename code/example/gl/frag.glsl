#version 330 core

in vec3 Color;
in vec2 TexCoord;

out vec4 FragColor;

uniform float Time;
uniform sampler2D TexKitten;
uniform sampler2D TexPuppy;

void main()
{
    vec4 ColKitten = texture(TexKitten, TexCoord);
    vec4 ColPuppy = texture(TexPuppy, TexCoord);
    
    if(TexCoord.y >= 0.5)
    {
        vec4 Water = vec4(0.7, 0.7, 1.0, 1.0);
        vec2 Sample = vec2((TexCoord.x + sin(TexCoord.y * 60.0 + Time * 2.0) / 30.0),
                           (1.0 - TexCoord.y));
        
        ColKitten = texture(TexKitten, Sample) * Water;
        ColPuppy = texture(TexPuppy, Sample) * Water;
    }
    
    float Mixing = ((sin(Time) + 1.0f)/2.0f);
    
    FragColor = mix(ColKitten, ColPuppy, Mixing); 
}