# !/bin/bash

export CC=gcc
export CXX=g++


ROOT_PWD=$(pwd)

# Go to temporary folder
mkdir -p tmp-onednn && cd tmp-onednn

# Clone oneDNN source code
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN

# Library is tested at this commit, but feel free to use different version
git checkout 9ef80d1732d054b7f12f0475d7181b37ffeba662

# Create build directory
mkdir -p build && cd build

# Configure CMake and generate makefiles
cmake .. -DCMAKE_INSTALL_PREFIX=${ROOT_PWD}/lib/onednn

# Build the library with half of the available cores to not overload the system.
make -j $((($(nproc) + 1) / 2 ))


# Install the library and headers
cmake --build . --target install

# Remove temporary folder
cd ${ROOT_PWD}
rm -rf tmp-onednn
