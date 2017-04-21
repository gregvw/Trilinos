
if(PYTHON_EXECUTABLE)

  unset(NUMPY_PATH)
  unset(NUMPY_INCLUDE_DIR)
  unset(NUMPY_VERSION)

  message("Looking for NumPy")
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_MODULE_PATH}/find_numpy.py 
                  OUTPUT_VARIABLE NUMPY_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NUMPY_OUTPUT)
    list( GET NUMPY_OUTPUT 0 NUMPY_INCLUDE_DIR )
    list( GET NUMPY_OUTPUT 1 NUMPY_VERSION     )
  endif()
    
  if(NUMPY_INCLUDE_DIR)
    message("Found NumPy version ${NUMPY_VERSION} in the path ${NUMPY_INCLUDE_DIR}")
    set(BUILD_WITH_NUMPY ON)
    add_definitions(-DENABLE_NUMPY=${CMAKE_ENABLE_NUMPY})
    include_directories(${NUMPY_INCLUDE_DIR})
  else()
    message("Could not find NumPy!")
  endif()

else()

  message(FATAL_ERROR "No Python interpreter set. Cannot find NumPy!")

endif()
