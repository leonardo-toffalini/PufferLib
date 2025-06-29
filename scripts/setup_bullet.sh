#!/bin/bash

git clone https://github.com/bulletphysics/bullet3.git
cd bullet3
mkdir build
cd build
cmake ..
make

