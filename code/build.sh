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

# Targets
hash=0
samples=0
cuversine=0
example=0
Targets="hash/samples/cuversine/example"

# Default
[ "$#" = 0 ] && example=1

for Arg in "$@"; do eval "$Arg=1"; done
# Exclusive flags
[ "$release" = 1 ] && debug=0

[ "$debug"   = 1 ] && printf '[debug mode]\n'
[ "$release" = 1 ] && printf '[release mode]\n'
mkdir -p "$Build"

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
 "
 DebugFlags="
	-g
	-lineinfo -src-in-ptx
	-DRL_INTERNAL=1
	"
 ReleaseFlags="-O3"

 [ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
 [ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"
 
 Flags="$Flags $WarningFlags"

 printf '%s\n' "$Source"
 Source="$(readlink -f "$Source")"
 
 $Compiler $Flags "$Source" -o "$Build"/"$Out"

 DidWork=1
}

C_Compile()
{
 Source="$1"
 Out="$2"

 Flags="${3:-}"

 [ "$gcc"   = 1 ] && Compiler="g++"
 [ "$clang" = 1 ] && Compiler="clang"
 printf '[%s compile]\n' "$Compiler"
 
 CommonCompilerFlags="-fsanitize-trap -nostdinc++ -fno-threadsafe-statics -I$ScriptDirectory"
 CommonWarningFlags="-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-sign-conversion -Wno-sign-compare -Wno-double-promotion -Wno-unused-but-set-variable -Wno-unused-variable -Wno-write-strings -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers"
 LinkerFlags=""

 DebugFlags="-g -ggdb -g3 -DRL_INTERNAL=1"
 ReleaseFlags="-O3"

 ClangFlags="-fdiagnostics-absolute-paths -ftime-trace
-Wno-null-dereference -Wno-missing-braces -Wno-vla-extension -Wno-writable-strings   -Wno-address-of-temporary -Wno-int-to-void-pointer-cast -Wno-reorder-init-list -Wno-c99-designator"

 GCCFlags="-Wno-cast-function-type -Wno-missing-field-initializers -Wno-int-to-pointer-cast"

 Flags="$CommonCompilerFlags $Flags"
 [ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
 [ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"
 Flags="$Flags $CommonWarningFlags"
 [ "$clang" = 1 ] && Flags="$Flags $ClangFlags"
 [ "$gcc"   = 1 ] && Flags="$Flags $GCCFlags"
 Flags="$Flags $LinkerFlags"

 printf '%s\n' "$Source"
 $Compiler $Flags "$(readlink -f "$Source")" -o "$Build"/"$Out"
 $Compiler $Flags "$(readlink -f "$Source")" -o "$Build"/"$Out"

 DidWork=1
}

Strip()
{
 Source="$1"
 Out="${1%.*}"
 Out="${Out##*/}"

 printf '%s %s' "$Source" "$Out" 
}

#- Compilation

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

if [ "$example" = 1 ]
then
 C_Compile ./example/ex_app.cpp app.so "-fPIC --shared -DBASE_NO_ENTRYPOINT=1" 
 C_Compile $(Strip ./example/ex_platform.cpp) "-lX11"
fi

#- End

if [ "$DidWork" = 0 ]
then
 printf 'ERROR: No valid build target provided.\n'
 printf 'Usage: %s <%s>\n' "$0" "$Targets"
else
 printf 'Done.\n' # 4coder bug
fi
