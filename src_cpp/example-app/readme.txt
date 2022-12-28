mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch .. # absolute path
cmake --build . --config Release
# make
