#!/bin/bash

mkdir -p build
cd build
make -j4
cd ..
build/Release/x86_64/bin/$1
