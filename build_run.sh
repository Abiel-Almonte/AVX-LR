#!/bin/bash
mkdir -p "build"
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=MinSizeRel -B "build"  #RelWithDebInfo, Debug, Release, MinSizeRel
cmake --build "build"
perf stat -e L1-dcache-loads,L1-dcache-load-misses ./build/run