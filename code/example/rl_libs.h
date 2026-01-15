#ifndef RL_LIBS_H
#define RL_LIBS_H

#if EX_SLOW_COMPILE
C_LINKAGE_BEGIN
# define GLAD_GL_IMPLEMENTATION
# define STB_IMAGE_IMPLEMENTATION
# define STB_TRUETYPE_IMPLEMENTATION
# define STB_SPRINTF_IMPLEMENTATION
# define RL_FONT_IMPLEMENTATION
#else
void GLADDisableCallbacks();
void GLADEnableCallbacks();
#endif

NO_WARNINGS_BEGIN
# include "lib/gl_core_3_3_debug.h"
# include "lib/stb_image.h"
# include "lib/stb_sprintf.h"
# include "lib/stb_truetype.h"
NO_WARNINGS_END

//~ GLAD helper functions

#if EX_SLOW_COMPILE

void GLADNullPreCallback(const char *name, GLADapiproc apiproc, int len_args, ...) {}
void GLADNullPostCallback(void *ret, const char *name, GLADapiproc apiproc, int len_args, ...) {}

void GLADDisableCallbacks()
{
    _pre_call_gl_callback = GLADNullPreCallback;
    _post_call_gl_callback = GLADNullPostCallback;
}

void GLADEnableCallbacks()
{
    _pre_call_gl_callback = _pre_call_gl_callback_default;
    _post_call_gl_callback = _post_call_gl_callback_default;
}

C_LINKAGE_END
#endif // EX_SLOW_COMPILE

#endif // RL_LIBS_H
