#ifndef RL_FONT_H
#define RL_FONT_H

#include <stdarg.h>
#include <stdio.h>
#include "lib/stb_truetype.h"

struct app_font
{
    stbtt_fontinfo Info;
    s32 Ascent;
    s32 Descent;
    s32 LineGap;
    v2 BoundingBox[2];
    b32 Initialized; // For debugging.
};

internal f32 rlf_GetBaseLine(app_font *Font, f32 Scale);
internal void rlf_InitFont(app_font *Font, char *FilePath);
internal void rlf_DrawCharacter(arena *Arena, app_offscreen_buffer *Buffer,  u8 *FontBitmap,
                                int FontWidth, int FontHeight, 
                                int XOffset, int YOffset,
                                u32 Color);
internal void rlf_DrawText(arena *Arena, app_offscreen_buffer *Buffer, app_font *Font, f32 HeightPixels,
                           str8 Text, v2 Offset, u32 Color, b32 IsUTF8);
internal void rlf_DrawTextInBox(arena *Arena, app_offscreen_buffer *Buffer, app_font *Font, 
                                str8 Text, f32 HeightPx, u32 Color,
                                v2 BoxMin, v2 BoxMax, b32 Centered);
internal void rlf_DrawTextFormat(arena *Arena, app_offscreen_buffer *Buffer, app_font *Font,
                                 f32 HeightPx, v2 Offset, u32 Color, b32 IsUTF8, 
                                 char *Format, ...);

//~ API

internal f32
rlf_GetBaseLine(app_font *Font, f32 Scale)
{
    f32 Result = Scale*(f32)(Font->Ascent - Font->Descent + Font->LineGap);
    return Result;
}

internal void
rlf_InitFont(app_font *Font, char *FilePath)
{
    str8 File = OS_ReadEntireFileIntoMemory(FilePath);
    
    if(File.Size)
    {
        if(stbtt_InitFont(&Font->Info, File.Data, stbtt_GetFontOffsetForIndex(File.Data, 0)))
        {
            Font->Info.data = (u8 *)File.Data;
            
            s32 X0, Y0, X1, Y1;
            stbtt_GetFontBoundingBox(&Font->Info, &X0, &Y0, &X1, &Y1);
            Font->BoundingBox[0] = v2{(f32)X0, (f32)Y0};
            Font->BoundingBox[1] = v2{(f32)X1, (f32)Y1};
            stbtt_GetFontVMetrics(&Font->Info, &Font->Ascent, &Font->Descent, &Font->LineGap);
            Font->Initialized = true;
        }
        else
        {
            // TODO(luca): Logging
        }
    }
    else
    {
        // TODO(luca): Logging
    }
}

internal f32
rlf_Lerp(f32 A, f32 B, f32 t)
{
    f32 Result = A*t + B*(1-t);
    return Result;
}

//- Rendering 
internal void
rlf_DrawCharacter(arena *Arena, app_offscreen_buffer *Buffer,  u8 *FontBitmap,
                  int FontWidth, int FontHeight, 
                  int XOffset, int YOffset,
                  u32 Color)
{
    s32 MinX = 0;
    s32 MinY = 0;
    s32 MaxX = FontWidth;
    s32 MaxY = FontHeight;
    
    if(XOffset < 0)
    {
        MinX = -XOffset;
        XOffset = 0;
    }
    if(YOffset < 0)
    {
        MinY = -YOffset;
        YOffset = 0;
    }
    if(XOffset + FontWidth > Buffer->Width)
    {
        MaxX -= ((XOffset + FontWidth) - Buffer->Width);
    }
    if(YOffset + FontHeight > Buffer->Height)
    {
        MaxY -= ((YOffset + FontHeight) - Buffer->Height);
    }
    
    u8 *Row = (u8 *)(Buffer->Pixels) + 
    (YOffset*Buffer->Pitch) +
    (XOffset*Buffer->BytesPerPixel);
    
    for(int  Y = MinY;
        Y < MaxY;
        Y++)
    {
        u32 *Pixel = (u32 *)Row;
        for(int X = MinX;
            X < MaxX;
            X++)
        {
            u8 Brightness = FontBitmap[Y*FontWidth+X];
            
            f32 Alpha = ((f32)Brightness/255.0f);
            
            f32 DR = (f32)((*Pixel >> 16) & 0xFF);
            f32 DG = (f32)((*Pixel >> 8) & 0xFF);
            f32 DB = (f32)((*Pixel >> 0) & 0xFF);
            
#if 0
            f32 R = rlf_Lerp(Color.R*255.0f, DR, Alpha);
            f32 G = rlf_Lerp(Color.G*255.0f, DG, Alpha);
            f32 B = rlf_Lerp(Color.B*255.0f, DB, Alpha);
            f32 A = 255.0f;
#else
            f32 SR = (f32)((Color >> 16) & 0xFF);
            f32 SG = (f32)((Color >> 8) & 0xFF);
            f32 SB = (f32)((Color >> 0) & 0xFF);
            
            f32 R = SR*Alpha;
            f32 G = SG*Alpha;
            f32 B = SB*Alpha;
            f32 A = Brightness;
#endif
            
            
            u32 Value = (((u32)(A) << 24) |
                         ((u32)(R) << 16) |
                         ((u32)(G) << 8) |
                         ((u32)(B) << 0));
            *Pixel = Value;
            
            Pixel += 1;
        }
        
        Row += Buffer->Pitch;
    }
}


internal void
rlf_DrawText(arena *Arena, app_offscreen_buffer *Buffer, app_font *Font, f32 HeightPixels,
             str8 Text, v2 Offset, u32 Color, b32 IsUTF8)
{
    Assert(Font->Initialized);
    
    Offset.X = roundf(Offset.X);
    Offset.Y = roundf(Offset.Y);
    
    f32 FontScale = stbtt_ScaleForPixelHeight(&Font->Info, HeightPixels);
    
    for EachIndex(Idx, Text.Size)
    {
        rune CharAt = (IsUTF8 ? ((rune *)Text.Data)[Idx] : Text.Data[Idx]);
        
        s32 FontWidth, FontHeight;
        s32 AdvanceWidth, LeftSideBearing;
        s32 X0, Y0, X1, Y1;
        u8 *FontBitmap = 0;
        stbtt_GetCodepointBitmapBox(&Font->Info, CharAt, 
                                    FontScale, FontScale, 
                                    &X0, &Y0, &X1, &Y1);
        FontWidth = (X1 - X0);
        FontHeight = (Y1 - Y0);
        FontBitmap = PushArray(Arena, u8, (FontWidth*FontHeight));
        stbtt_MakeCodepointBitmap(&Font->Info, FontBitmap, 
                                  FontWidth, FontHeight, FontWidth, 
                                  FontScale, FontScale, CharAt);
        
        stbtt_GetCodepointHMetrics(&Font->Info, CharAt, &AdvanceWidth, &LeftSideBearing);
        
        s32 XOffset = floorf(Offset.X + LeftSideBearing*FontScale);
        s32 YOffset = Offset.Y + Y0;
        
        rlf_DrawCharacter(Arena, Buffer, FontBitmap, FontWidth, FontHeight, XOffset, YOffset, Color);
        
        Offset.X += roundf(AdvanceWidth*FontScale);
    }
}

// 1. First pass where we check each character's size.
// 2. Save positions where we need to wrap.
// Wrapping algorithm
// 1. When a character exceeds the box maximum width search backwards for whitespace.
//    I. Whitespace found?
//       Y -> Length of string until whitespace would fit?
//            Y -> Save whitespace's position.  This becomes the new searching start position.
//            N -> Break on the character that exceeds the maximum width.
//       N -> Break on the character that exceeds the maximum width.
//    II. Continue until end of string.

internal void
rlf_DrawTextInBox(arena *Arena, app_offscreen_buffer *Buffer, app_font *Font, 
                  str8 Text, f32 HeightPx, u32 Color,
                  v2 BoxMin, v2 BoxMax, b32 Centered)
{
    s32 *CharacterPixelWidths = PushArray(Arena, s32, Text.Size);
    u32 *WrapPositions = PushArray(Arena, u32, 0);
    u32 WrapPositionsCount = 0;
    
    f32 FontScale = stbtt_ScaleForPixelHeight(&Font->Info, HeightPx);
    
    // TODO(luca): UTF8 support
    // e.g. (https://en.wikipedia.org/wiki/Whitespace_character) these are all whitespace characters that we might want to support.
    for(u32 TextIndex = 0;
        TextIndex < Text.Size;
        TextIndex++)
    {
        u8 CharAt = Text.Data[TextIndex];
        
        s32 AdvanceWidth, LeftSideBearing;
        stbtt_GetCodepointHMetrics(&Font->Info, CharAt, &AdvanceWidth, &LeftSideBearing);
        
        CharacterPixelWidths[TextIndex] = roundf(FontScale*AdvanceWidth);
    }
    
    s32 MaxWidth = BoxMax.X - BoxMin.X;
    Assert(MaxWidth >= 0);
    
    u32 SearchStart = 0;
    while(SearchStart < Text.Size)
    {
        s32 CumulatedWidth = 0;
        u32 SearchIndex = SearchStart;
        for(;
            ((SearchIndex < Text.Size) &&
             (CumulatedWidth <= MaxWidth));
            SearchIndex++)
        {
            s32 Width = CharacterPixelWidths[SearchIndex];
            CumulatedWidth += Width;
        }
        
        if(CumulatedWidth > MaxWidth)
        {
            // We need to search backwards for wrapping.
            SearchIndex--;
            u32 SearchIndexStop = SearchIndex;
            
            while(SearchIndex > SearchStart)
            {
                if(Text.Data[SearchIndex] == ' ')
                {
                    PushStruct(Arena, u32);
                    WrapPositions[WrapPositionsCount++] = SearchIndex;
                    break;
                }
                
                SearchIndex--;
            }
            
            if(SearchIndex > SearchStart)
            {
                Assert(SearchIndex > SearchStart);
                // luca: We skip the character we wrapped on.
                SearchStart = SearchIndex + 1;
            }
            else if(SearchIndex == SearchStart)
            {
                Assert(SearchIndexStop > SearchStart);
                PushStruct(Arena, u32);
                WrapPositions[WrapPositionsCount++] = SearchIndexStop;
                SearchStart = SearchIndexStop;
            }
            else
            {
                Assert(0);
            }
            
        }
        else
        {
            // luca: We don't need to wrap, we've reached the end of the text.
            break;
        }
    }
    
    f32 YAdvance = rlf_GetBaseLine(Font, FontScale);
    v2 TextOffset = BoxMin;
    TextOffset.Y += YAdvance;
    
    if(Centered)
    {
        s32 TextHeight = YAdvance * (WrapPositionsCount + 1);
        s32 CenterHOffset = ((BoxMax.Y - BoxMin.Y) - TextHeight)/2;
        if(CenterHOffset >= 0)
        {
            TextOffset.Y += CenterHOffset;
        }
    }
    
    u32 Start = 0;
    for(u32 WrapIndex = 0;
        WrapIndex < WrapPositionsCount;
        WrapIndex++)
    {
        u32 Position = WrapPositions[WrapIndex];
        
        if(TextOffset.Y - FontScale*Font->Descent < BoxMax.Y)
        {
            
            b32 DoCenter = (Centered && 
                            ((WrapIndex == 0) ||
                             (Text.Data[(WrapPositions[WrapIndex - 1])] == ' ')));
            if(DoCenter)
            {
                s32 TextWidth = 0;
                for(u32 WidthIndex = Start;
                    WidthIndex < Position;
                    WidthIndex++)
                {
                    TextWidth += CharacterPixelWidths[WidthIndex];
                }
                TextOffset.X = BoxMin.X + ((MaxWidth - TextWidth)/2);
            }
            
            rlf_DrawText(Arena, Buffer, Font, HeightPx, 
                         str8{Text.Data + Start, Position - Start}, 
                         TextOffset, Color, false);
        }
        
        TextOffset.Y += YAdvance;
        
        if(Text.Data[Position] == ' ')
        {
            Position++;
        }
        
        Start = Position;
    }
    
    TextOffset.X = BoxMin.X;
    b32 DoCenter = (Centered &&
                    ((WrapPositionsCount == 0) || (Text.Data[WrapPositions[WrapPositionsCount - 1]] == ' ')));
    if(DoCenter)
    {                
        s32 TextWidth = 0;
        for EachIndex(Idx, Text.Size)
        {
            TextWidth += CharacterPixelWidths[Idx];
        }
        TextOffset.X = BoxMin.X + ((MaxWidth - TextWidth)/2);
    }
    
    if(TextOffset.Y - FontScale*Font->Descent < BoxMax.Y)
    {
        rlf_DrawText(Arena, Buffer, Font, HeightPx, 
                     str8{Text.Data + Start, Text.Size - Start}, 
                     TextOffset, Color, false); 
    }
}

internal void
rlf_DrawTextFormat(arena *Arena, app_offscreen_buffer *Buffer, app_font *Font,
                   f32 HeightPx, v2 Offset, u32 Color, b32 IsUTF8, 
                   char *Format, ...)
{
    str8 Text = {0};
    Text.Data = PushArray(Arena, u8, 256);
    va_list Args;
    va_start(Args, Format);
    Text.Size = (umm)vsprintf((char *)Text.Data, Format, Args);
    
    rlf_DrawText(Arena, Buffer, Font, HeightPx, Text, Offset, Color, IsUTF8);
}

#endif //RL_FONT_H