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

# Default
[ "$#" = 0 ] && hash=1

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
 -I$ScriptDirectory -DOS_LINUX=1 -DAOC_INTERNAL=1
 -arch sm_50
 "
 WarningFlags="
 -diag-suppress 1143
 -diag-suppress 2464
 -diag-suppress 177
 -diag-suppress 550
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
 DebugFlags="-g -G"
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

 [ "$gcc"   = 1 ] && Compiler="g++"
 [ "$clang" = 1 ] && Compiler="clang"
 printf '[%s compile]\n' "$Compiler"
 
 CommonCompilerFlags="-DOS_LINUX=1 -fsanitize-trap -nostdinc++ -I$ScriptDirectory"
 CommonWarningFlags="-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-sign-conversion -Wno-sign-compare -Wno-double-promotion -Wno-unused-but-set-variable -Wno-unused-variable -Wno-write-strings -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers"
 LinkerFlags=""

 DebugFlags="-g -ggdb -g3 -DAOC_INTERNAL=1"
 ReleaseFlags="-O3"

 ClangFlags="-fdiagnostics-absolute-paths -ftime-trace
-Wno-null-dereference -Wno-missing-braces -Wno-vla-extension -Wno-writable-strings   -Wno-address-of-temporary -Wno-int-to-void-pointer-cast"

 GCCFlags="-Wno-cast-function-type -Wno-missing-field-initializers -Wno-int-to-pointer-cast"

 Flags="$CommonCompilerFlags"
 [ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
 [ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"
 Flags="$Flags $CommonWarningFlags"
 [ "$clang" = 1 ] && Flags="$Flags $ClangFlags"
 [ "$gcc"   = 1 ] && Flags="$Flags $GCCFlags"
 Flags="$Flags $LinkerFlags"

 printf '%s\n' "$Source"
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

[ "$hash"    = 1 ] && C_Compile  $(Strip ./hash/hash.c)
if [ "$samples" = 1 ]
then
	ls lib
 CU_Compile $(Strip ./lib/cuda-samples/deviceQuery.cpp)
 CU_Compile $(Strip ./lib/cuda-samples/deviceQueryDrv.cpp) -lcuda
 CU_Compile $(Strip ./lib/cuda-samples/topologyQuery.cu)
fi

if [ "$DidWork" = 0 ]
then
 printf 'ERROR: No valid build target provided.\n'
 printf 'Usage: %s <samples/day1/day2/day3/day3_cu>\n' "$0"
fi
