set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pyrol/__init__.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pyrol/test/__init__.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

message("Running setup.py with Python interpreter: ${PYTHON_EXECUTABLE}")

configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS ${DEPS})

if(PYROL_BUILD_INPLACE)
  install(CODE "execute_process(COMMAND  ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace)")
else()
  install(CODE "execute_process(COMMAND  ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=${PYROL_INSTALL})")
endif()
