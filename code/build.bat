@echo off

call C:\msvc\setup_x64.bat

cd %~dp0

IF NOT EXIST ..\build mkdir ..\build
pushd ..\build

raddbg --ipc kill

set CommonLinkerFlags=-opt:ref -incremental:no user32.lib Gdi32.lib winmm.lib Opengl32.lib

set CommonCompilerFlags=-MTd -Gm- -nologo -GR- -EHa- -Oi -FC -Z7 -WX -W4 -wd4459 -wd4456 -wd4201 -wd4100 -wd4101 -wd4189 -wd4505 -wd4996 -wd4389 -wd4244 -DRL_INTERNAL=1 -I..\code /std:c++20 /Zc:strictStrings-

REM 64-Bit

echo WAITING FOR PDB > lock.tmp

cl %CommonCompilerFlags% -Fmapp.map ..\code\example\ex_app.cpp -LD /link /DLL %CommonLinkerFlags% /EXPORT:UpdateAndRender
del lock.tmp

cl %CommonCompilerFlags% -Fmex_platform.map ..\code\example\ex_platform.cpp /link %CommonLinkerFlags%