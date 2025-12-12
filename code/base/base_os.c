#if OS_LINUX
# include "base_linux.c"
#elif OS_WINDOWS
# include "base_windows.c"
#else 
# error "Operating system not provided or supported."
#endif
