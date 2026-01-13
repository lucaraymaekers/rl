/* date = January 13th 2026 8:59 am */

#ifndef EX_APP_H
#define EX_APP_H

NO_WARNINGS_BEGIN
#include "ex_font.h"
NO_WARNINGS_END

typedef unsigned int gl_handle;

typedef s32 face[4][3];

typedef struct button button;
struct button
{
    str8 Text;
    v2 Min;
    v2 Max;
    f32 Radius;
};


typedef struct button_render_data button_render_data;
struct button_render_data
{
    v3 *Vertices;
    v3 *Colors;
    v2 *Minima;
    v2 *Maxima;
    f32 *Radii;
};

typedef struct model_path model_path;
struct model_path
{
    str8 Name;
    str8 Model;
    str8 Texture;
};

struct gl_render_data
{
    s32 Major, Minor;
    gl_handle VAO;
    gl_handle VBO[8];
    gl_handle Tex[2];
    gl_handle ModelShader, TextShader, ButtonShader;
};

typedef struct app_state app_state;
struct app_state
{
    random_series Series;
    
    app_font Font;
    
    gl_render_data Render;
    
    v3 Offset;
    v2 Angle;
    b32 Animate;
    
    s32 SelectedModelIdx;
};


#endif //EX_APP_H
