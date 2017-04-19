# Example script of configuring PyROL with user choice of Python Version
# This script should be executed in a subdirectory, e.g. /build, of the
# PyROL source directory

VERSION="2.7"
PYTHON_BASE_PATH="/usr/local/"
PYMALLOC_BUILD=false
TRILINOS_INSTALL="${USER_HOME}/Projects/PyROL/trilinos_install"
C_COMPILER=`which gcc`
CXX_COMPILER=`which g++`
CXXFLAGS="-std=c++11 -fPIC -g"

# This should work for Linux and Mac
USER_HOME=`eval echo "~$USER"`

OS=`uname`

if [ $OS == "Darwin" ]; then
  EXT=".dynlib"
else
  EXT=".so"
fi

CURRENT_DIR=`pwd`
PYROL_HOME=`dirname ${CURRENT_DIR}`

if [ $PYMALLOC_BUILD == true ]; then
  PYTHON_NAME="python${VERSION}m"
else
  PYTHON_NAME="python${VERSION}"
fi

PYTHON_INCLUDE_DIR="${PYTHON_BASE_PATH}/include/${PYTHON_NAME}"
PYTHON_LIBRARY="${PYTHON_BASE_PATH}/lib/lib${PYTHON_NAME}${EXT}"
PYTHON_INTERPRETER="${PYTHON_BASE_PATH}/bin/${PYTHON_NAME}"
 

cmake . \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -D CMAKE_VERBOSE_COMPILE:BOOL=ON \
  -D CMAKE_C_COMPILER:FILEPATH=${C_COMPILER} \
  -D CMAKE_CXX_COMPILER:FILEPATH=${CXX_COMPILER} \
  -D Trilinos_PREFIX:PATH=${TRILINOS_INSTALL} \
  -D PYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR} \
  -D PYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY} \
  -D PYTHON_INTERPRETER:FILEPATH=${PYTHON_INTERPRETER} \
${PYROL_HOME}
