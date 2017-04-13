cmake_minimum_required(VERSION 3.1)

project(PYROL)

if(NOT PYROL_INSTALL)
  set(PYROL_INSTALL "${CMAKE_SOURCE_DIR}/bin")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(USER_SPECIFIED_PYTHON (PYTHON_LIBRARY AND PYTHON_INCLUDE_DIR) )

if(USER_SPECIFIED_PYTHON)
  message("Using specified Python")
else()


  message("Searching for Python...")
  message("If you want to specify the installation of Python to use, set the cache variables:")
  message("PYTHON_LIBRARY      - path to the python library")
  message("PYTHON_INCLUDE_DIR  - path to Python.h")

  find_package(PythonInterp 2.7 REQUIRED)
  find_package(PythonLibs 2.7 REQUIRED)

  if( NOT PythonInterp_FOUND ) 
    message(FATAL_ERROR "Could not find Python!")
  else()
    message("Found Python interpreter: ${PYTHON_EXECUTABLE}")
  endif()

  if( NOT PythonLibs_FOUND )
    message(FATAL_ERROR "Could not find Python library!")
  else()
    message("Found Python library: ${PYTHON_LIBRARIES}")
  endif()

 if( NOT (PYTHON_VERSION_STRING STREQUAL PYTHONLIBS_VERSION_STRING) )
   message(FATAL_ERROR "The Python interpreter and libraries have incompatible!")
  endif()

endif(USER_SPECIFIED_PYTHON)


include_directories( ${PYTHON_INCLUDE_DIRS} )
link_directories( ${PYTHON_LIBRARIES} )

add_subdirectory(src)
add_subdirectory(python)




