#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "You must supply the Trilinos source path"
  exit 1
fi

# This should work for Linux and Mac
USER_HOME=`eval echo "~$USER"`

TRILINOS_HOME=$1 

# Modify these paths to point to where you want to 
TRILINOS_BUILD="${USER_HOME}/projects/PyROL/trilinos_build"
TRILINOS_INSTALL="${USER_HOME}/projects/PyROL/trilinos_install"

echo "Trilinos build directory: ${TRILINOS_BUILD}"
echo "Trilinos install directory ${TRILINOS_INSTALL}"

CXXFLAGS="-std=c++11 -fPIC"

if [ ! -d $TRILINOS_BUILD ]; then
  mkdir -p $TRILINOS_BUILD
  cd $TRILINOS_BUILD
else
  cd $TRILINOS_BUILD
  if [ -f CMakeCache.txt ]; then
    rm CMakeCache.txt
  fi
  if [ -d CMakeFiles ]; then
    rm -rf CMakeFiles
  fi
fi

cmake ..\
 -D CMAKE_INSTALL_PREFIX:PATH=${TRILINOS_INSTALL} \
 -D CMAKE_BUILD_TYPE:STRING=DEBUG \
 -D BUILD_SHARED_LIBS:BOOL=ON \
\
 -D TPL_ENABLE_MPI:BOOL=OFF \
\
 -D Trilinos_INSTALL_LIBRARIES_AND_HEADERS:BOOL=ON \
 -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_TESTS:BOOL=OFF \
 -D Trilinos_ENABLE_ROL:BOOL=ON \
 -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
${TRILINOS_HOME}


