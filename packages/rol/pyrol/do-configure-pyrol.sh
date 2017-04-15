# Example script of configuring PyROL with user choice of Python Version
# This script should be executed in a subdirectory, e.g. /build, of the
# PyROL source directory

VERSION="3.3"
PYMALLOC_BUILD=true   

OPT_PYTHON="/opt/local/Library/Frameworks/Python.framework/Versions/${VERSION}"


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

PYTHON_INCLUDE_DIR="${OPT_PYTHON}/include/${PYTHON_NAME}"
PYTHON_LIBRARY="${OPT_PYTHON}/lib/lib${PYTHON_NAME}${EXT}"

cmake .. \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -D PYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR} \
  -D PYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY} \
${PYROL_HOME}
