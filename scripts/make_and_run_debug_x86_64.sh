#!/bin/bash

mkdir -p build
cd build
make -j4
cd ..
build/Debug/x86_64/bin/$1
