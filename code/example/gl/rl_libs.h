#ifndef RL_LIBS_H
#define RL_LIBS_H

#include "base/base_core.h"

C_LINKAGE_BEGIN

#if RL_FAST_COMPILE
// NOTE(luca): If fast compile mode is on, the implementation will be compiled to a separate translation unit (see `../build/rl_libs.o` in `build.sh`).
//
# include "lib/gl_core_3_3_debug.h"
# include "lib/stb_image.h"
# include "lib/stb_truetype.h"
# include "lib/stb_sprintf.h"
void GLADDisableCallbacks();
void GLADEnableCallbacks();

#else

NO_WARNINGS_BEGIN
# define GLAD_GL_IMPLEMENTATION
# define STB_IMAGE_IMPLEMENTATION
# define STB_TRUETYPE_IMPLEMENTATION
# define STB_SPRINTF_IMPLEMENTATION
# define RL_FONT_IMPLEMENTATION
# include "lib/gl_core_3_3_debug.h"
# include "lib/stb_image.h"
# include "lib/stb_sprintf.h"
# include "lib/stb_truetype.h"
NO_WARNINGS_END

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
#endif // RL_FAST_COMPILE

C_LINKAGE_END
#endif // RL_LIBS_H
