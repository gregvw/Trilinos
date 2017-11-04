# Example script of configuring PyROL with user choice of Python Version
# This script should be executed in a subdirectory, e.g. /build, of the
# PyROL source directory

VERSION="3.3"
PYMALLOC_BUILD=true   

PYTHON_BASE_PATH="/opt/local/Library/Frameworks/Python.framework/Versions/${VERSION}"

# This should work for Linux and Mac
USER_HOME=`eval echo "~$USER"`

TRILINOS_HOME=$1 

TRILINOS_INSTALL="${USER_HOME}/Projects/PyROL/trilinos_install"

CXXFLAGS="-std=c++11 -fPIC"

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
 
if [ -f CMakeCache.txt ]; then
  rm CMakeCache.txt
fi
if [ -d CMakeFiles ]; then
  rm -rf CMakeFile
fi

#  -D PYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR} \
#  -D PYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY} \
#  -D PYTHON_INTERPRETER:FILEPATH=${PYTHON_INTERPRETER} \

cmake . \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -D CMAKE_CXX_FLAGS:STRING=${CXXFLAGS} \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -D Trilinos_PREFIX:PATH=${TRILINOS_INSTALL} \
${PYROL_HOME}
