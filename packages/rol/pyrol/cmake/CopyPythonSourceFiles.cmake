file( COPY setup.py src test bin DESTINATION "${CMAKE_ARGV3}"
  FILES_MATCHING PATTERN "*.py" )
