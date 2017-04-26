
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pyrol/__init__.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

message("Running setup.py with Python interpreter: ${PYTHON_EXECUTABLE}")

set(BUILD_EXT "build_ext")
string(CONCAT BUILD_EXT ${BUILD_EXT} " --debug")
string(CONCAT BUILD_EXT ${BUILD_EXT} " --build-lib=${PYROL_TEST}")
string(CONCAT BUILD_EXT ${BUILD_EXT} " --rpath=${Trilinos_LIBRARY_DIRS}")


message("BUILD_EXT=${BUILD_EXT}")

configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY}
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS ${DEPS})

install(CODE "execute_process(COMMAND  ${PYTHON_EXECUTABLE} ${SETUP_PY} ${BUILD_EXT})")
