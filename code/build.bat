@echo off

call C:\msvc\setup_x64.bat

cd %~dp0

IF NOT EXIST ..\build mkdir ..\build
pushd ..\build


set CommonLinkerFlags=-opt:ref -incremental:no user32.lib Gdi32.lib winmm.lib

cl -MTd -Gm- -nologo -GR- -EHa- -Oi -FC -Z7 -WX -W4 -wd4459 -wd4456 -wd4201 -wd4100 -wd4101 -wd4189 -wd4505 -wd4996 -wd4389 -wd4244 -DRL_INTERNAL=1 -I..\code /std:c++20 ..\code\example\platform.cpp /link %CommonLinkerFlags%