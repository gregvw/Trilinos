cmake_minimum_required(VERSION 3.1)

project(PYROL)
message("Configuring PyROL")

set(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)



set(Trilinos_DIR ${Trilinos_INSTALL})
set(Trilinos_SRC_DIR ${Trilinos_SRC_DIR})

set( Trilinos_TPL_INCLUDE_DIRS ${Trilinos_TPL_INCLUDE_DIRS} )
set( Trilinos_TPL_LIBRARY_DIRS ${Trilinos_TPL_LIBRARY_DIRS} )

find_package(Trilinos REQUIRED)

if(NOT Trilinos_FOUND) 
  message(FATAL_ERROR "Could not find Trilinos!")
endif()

message("\nFound Trilinos! Here are details: ")
message("   Trilinos_VERSION = ${Trilinos_VERSION}")
message("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
message("   Trilinos_LIBRARIES  = ${Trilinos_LIBRARIES}")
message("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
message("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
message("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
message("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
message("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
message("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
message("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")

find_package(Python)


if(NOT PYROL_INSTALL)
  set(PYROL_INSTALL "${CMAKE_SOURCE_DIR}/bin")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# These flags where needed to build on OSX. 
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-lpython,-undefined,dynamic_lookup")
endif()

include_directories( "${PYROL_SOURCE_DIR}/include" )
include_directories( "${PYROL_SOURCE_DIR}/include/test" )
include_directories( ${Trilinos_INCLUDE_DIRS} )

link_directories( ${Trilinos_LIBRARY_DIRS} )


message("PYTHON = ${PYTHON_EXECUTABLE}")

add_subdirectory(src)
add_subdirectory(python)




