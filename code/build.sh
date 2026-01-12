#!/bin/sh

set -eu

ScriptDirectory="$(dirname "$(readlink -f "$0")")"
cd "$ScriptDirectory"

#- Main
DidWork=0
Build="../build"

clang=1
gcc=0
debug=1
release=0
personal=0
nofast=0
fast=1

# Targets
clean=0
all=0
hash=0
samples=0
cuversine=0
example=0
app=0
sort=0
gl=0
windows=0
cling=0
rldroid=0
Targets="hash/samples/cling/rldroid/cuversine/example [sort/app/gl/windows]\n"

# Default
[ "$#" = 0 ] && example=1 && app=1

for Arg in "$@"; do eval "$Arg=1"; done
# Exclusive flags
[ "$release" = 1 ] && debug=0
[ "$gcc"     = 1 ] && clang=0
[ "$nofast"  = 1 ] && fast=0
mkdir -p "$Build"

[ -f "./base/base_build.h" ] && personal=1

CU_Compile()
{
 Source="$1"
 Out="$2"

 Flags="${3:-}"

 Compiler=nvcc
 printf '[%s compile]\n' "$Compiler"

 Flags="$Flags
 -I$ScriptDirectory
 --threads 0
 --use_fast_math
 --generate-code arch=compute_60,code=sm_60
 --resource-usage
 --time $Build/${Out}_time.txt
 "

 WarningFlags="
 -diag-suppress 1143
 -diag-suppress 2464
 -diag-suppress 177
 -diag-suppress 550
 -diag-suppress 114
 -Wno-deprecated-gpu-targets
 -Xcompiler -Wall
 -Xcompiler -Wextra
 -Xcompiler -Wconversion
 -Xcompiler -Wdouble-promotion

 -Xcompiler -Wno-pointer-arith
 -Xcompiler -Wno-attributes 
 -Xcompiler -Wno-unused-but-set-variable 
 -Xcompiler -Wno-unused-variable 
 -Xcompiler -Wno-write-strings
 -Xcompiler -Wno-pointer-arith
 -Xcompiler -Wno-unused-parameter
 -Xcompiler -Wno-unused-function
 -Xcompiler -Wno-missing-field-initializers
 -Xcompiler -Wno-sign-compare
 "
 DebugFlags="-g -lineinfo -src-in-ptx"
 ReleaseFlags="-O3"

 [ "$debug"    = 1 ] && Flags="$Flags $DebugFlags"
 [ "$release"  = 1 ] && Flags="$Flags $ReleaseFlags"
 [ "$personal" = 1 ] && Flags="$Flags -DRL_PERSONAL=1" 
 Flags="$Flags $WarningFlags"

 printf '%s\n' "$Source"
 Source="$(readlink -f "$Source")"
 
 $Compiler $Flags "$Source" -o "$Build"/"$Out"

 DidWork=1
}

C_Compile()
{
 SourceFiles="$1"
 Out="$2"

 Flags="${3:-}"

 # NOTE(luca): _GNU_SOURCE is only for C source files since it is enabled by default in c++.
 CommonCompilerFlags="-fsanitize-trap -nostdinc++ -fno-threadsafe-statics -I$ScriptDirectory -D_GNU_SOURCE=1"
 CommonWarningFlags="-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-sign-conversion -Wno-sign-compare -Wno-double-promotion -Wno-unused-but-set-variable -Wno-unused-variable -Wno-write-strings -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers"
 LinkerFlags="-lm -ldl -lpthread"

 DebugFlags="-g -ggdb -g3"
 ReleaseFlags="-O3"

 ClangFlags="-fdiagnostics-absolute-paths -fsanitize-undefined-trap-on-error -ftime-trace
-Wno-null-dereference -Wno-missing-braces -Wno-vla-extension -Wno-writable-strings   -Wno-address-of-temporary -Wno-int-to-void-pointer-cast -Wno-reorder-init-list -Wno-c99-designator"

 GCCFlags="-Wno-cast-function-type -Wno-missing-field-initializers -Wno-int-to-pointer-cast"

 Flags="$CommonCompilerFlags $Flags"
 [ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
 [ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"
 [ "$personal" = 1 ] && Flags="$Flags -DRL_PERSONAL=1"
 Flags="$Flags $CommonWarningFlags"
 [ "$clang" = 1 ] && Flags="$Flags $ClangFlags"
 [ "$gcc"   = 1 ] && Flags="$Flags $GCCFlags"
 Flags="$Flags $LinkerFlags"

 printf '%s\n' "$SourceFiles"

 Source=
 for File in $SourceFiles
 do Source="$Source $(readlink -f "$File")"
 done
 
 $Compiler $Flags $Source -o "$Build"/"$Out"

 DidWork=1
}

Strip()
{
 Source="$1"
 Out="${1%.*}"
 Out="${Out##*/}"

 printf '%s %s' "$Source" "$Out" 
}

#- Targets
if [ "$clean"  = 1 ]
then
 rm -rf ../build/*
 DidWork=1
fi

[ "$debug"   = 1 ] && printf '[debug mode]\n'
[ "$release" = 1 ] && printf '[release mode]\n'
[ "$gcc"   = 1 ] && Compiler="g++"
[ "$clang" = 1 ] && Compiler="clang"
printf '[%s compile]\n' "$Compiler"

[ "$hash" = 1 ] && C_Compile $(Strip ./hash/hash.c)
if [ "$samples" = 1 ]
then
 CU_Compile $(Strip ./cuda-samples/deviceQuery.cpp)
 CU_Compile $(Strip ./cuda-samples/deviceQueryDrv.cpp) -lcuda
 CU_Compile $(Strip ./cuda-samples/topologyQuery.cu)
fi

if [ "$cuversine" = 1 ]
then
 CU_Compile ./cuversine/app.cu app.so "--compiler-options '-fPIC' --shared" 
 CU_Compile $(Strip ./cuversine/platform.cpp) "-lX11"
fi

AppCompile()
{
 Dir="$1"
	ExtraFlags="${2:-}"

 AppFlags="-fPIC --shared" 

 LibsFile="../build/rl_libs.o"
 if [ "$fast" = 1 ]
 then
  [ ! -f "$LibsFile" ] && C_Compile "$Dir"/rl_libs.h "$LibsFile" "-fPIC -x c++ -c -Wno-unused-command-line-argument"
  AppFlags="$AppFlags -DRL_FAST_COMPILE=1 $LibsFile"
 fi
 C_Compile "$Dir"/ex_app.cpp ex_app.so "$AppFlags $ExtraFlags"
 C_Compile $(Strip $Dir/ex_platform.cpp) "-lX11 -lGL -lGLX $ExtraFlags"
}

if [ "$example" = 1 ]
then
 [ "$app"  = 1 ] && AppCompile ./example
 [ "$sort" = 1 ] && AppCompile ./example/sort "-DEX_FORCE_X11=1"
 [ "$gl"   = 1 ] && AppCompile ./example/gl
	if [ "$windows" = 1 ]
	then
		printf '[debug mode]\n'
		printf '[windows compile]\n'
		printf 'call C:\BuildTools\devcmd.bat\ncall build.bat\n' | wine cmd.exe 2>/dev/null
		DidWork=1
	fi
fi

if [ "$cling" = 1 ]
then 
 [ ! -f "../build/cling" ] && 
		clang -fdiagnostics-absolute-paths -D_GNU_SOURCE -Wno-writable-strings -I. -g -o ../build/cling ./cling/example.c
	cd ..
	./build/cling
 DidWork=1
fi

if [ "$rldroid" = 1 ] 
then
 cd ./lib/rawdrawandroid/
 make -B push run
 DidWork=1
fi

#- End

if [ "$DidWork" = 0 ]
then
 printf 'ERROR: No valid build target provided.\n'
 printf 'Usage: %s <%s>\n' "$0" "$Targets"
else
 printf 'Done.\n' # 4coder bug
fi
