#!/bin/bash

mkdir -p build
cd build
rm -rf ./*
cmake $@ ..
make -j4
cd ..
