set(USER_SPECIFIED_PYTHON FALSE)

if( PYTHON_INTERPRETER )
  message("PYTHON_INTERPRETER = ${PYTHON_INTERPRETER}")
  if( PYTHON_INCLUDE_DIR )
    message("PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")
    if( PYTHON_LIBRARY ) 
       message("PYTHON_LIBRARY = ${PYTHON_LIBRARY}")
      set(USER_SPECIFIED_PYTHON TRUE)
    endif()
  endif()
endif()


if(USER_SPECIFIED_PYTHON)
  message("Using specified Python")
  
  include_directories(${PYTHON_INCLUDE_DIR})
  get_filename_component(PYTHON_LIBRARY_DIR ${PYTHON_LIBRARY} DIRECTORY)
  link_directories(${PYTHON_LIBRARY_DIR})

  set( PYTHON_EXECUTABLE ${PYTHON_INTERPRETER} )
else()

  message("Searching for Python...")
  message("If you want to specify the installation of Python to use, set the cache variables:")
  message("PYTHON_LIBRARY      - path to the python library")
  message("PYTHON_INCLUDE_DIR  - path to Python.h")

  find_package(PythonInterp 2.7 REQUIRED)
  find_package(PythonLibs 2.7 REQUIRED)

  if( NOT PYTHONINTERP_FOUND ) 
    message(FATAL_ERROR "Could not find Python!")
  else()
    message("Found Python interpreter: ${PYTHON_EXECUTABLE}")
  endif()

  if( NOT PYTHONLIBS_FOUND )
    message(FATAL_ERROR "Could not find Python library!")
  else()
    message("Found Python library: ${PYTHON_LIBRARIES}")
  endif()

 # message("PYTHON_VERSION_STRING = ${PYTHON_VERSION_STRING}")
 
 # If the automated search for Python interpreter and libraries
 # return incompatible versions, use the higher version number
 if( NOT (PYTHON_VERSION_STRING STREQUAL PYTHONLIBS_VERSION_STRING) )
   message("The Python interpreter and libraries have have different versions")
    
   string( REPLACE "." ";" PYINTERP_VERSION_LIST ${PYTHON_VERSION_STRING} )
   string( REPLACE "." ";" PYLIBS_VERSION_LIST ${PYTHONLIBS_VERSION_STRING} )
  
   list( GET PYINTERP_VERSION_LIST 0 PYINTERP_VERSION_MAJOR )
   list( GET PYINTERP_VERSION_LIST 1 PYINTERP_VERSION_MINOR )
   list( GET PYINTERP_VERSION_LIST 2 PYINTERP_VERSION_PATCH )
   list( GET PYLIBS_VERSION_LIST 0 PYLIBS_VERSION_MAJOR )
   list( GET PYLIBS_VERSION_LIST 1 PYLIBS_VERSION_MINOR )
   list( GET PYLIBS_VERSION_LIST 2 PYLIBS_VERSION_PATCH )

   if( NOT (PYINTERP_VERSION_MAJOR EQUAL PYLIBS_VERSION_MAJOR) )
     message( FATAL_ERROR "The major versions do not match" )
   endif()
   if( NOT (PYINTERP_VERSION_MINOR EQUAL PYLIBS_VERSION_MINOR) )
     message( FATAL_ERROR "The minor versions do not match" )
   endif()
   if( NOT(PYINTERP_VERSION_PATCH EQUAL PYLIBS_VERSION_PATCH) )
     message( "The Python major and minor versions match, however, the patch ")
     message( "versions do not match. This build may not work correctly. It ")
     message( "is recommended to specify which Python interpreter and library ")
     message( "to use" )
   endif()
   set(PY_VERSION "${PYLIBS_VERSION_MAJOR}.${PYLIBS_VERSION_MINOR}")  
 endif()
 
 include_directories( ${PYTHON_INCLUDE_DIRS} )
 link_directories( ${PYTHON_LIBRARIES} )

endif(USER_SPECIFIED_PYTHON)

