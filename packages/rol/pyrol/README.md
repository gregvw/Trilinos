PyROL
=====

PyROL is a Python front-end to the Rapid Optimization Library (ROL)


Required Build Dependencies
---------------------------

- Trilinos 
- Python 2.7 or greater
- CMake 3.1 or greater


Optional Build Dependencies
---------------------------

- NumPy
- Dolfin


Trilinos Build Instructions
---------------------------

- Edit do-configure-trilinos.sh to specify where you want to build and
  install Trilinos

- Execute the script with ./do-configure-trilinos

- From the TRILINOS build directory call make -jN, where N is the number of 
  cores you want to use and the make install


PyROL Build Instructions
------------------------

Make a subdirectory of the PyROL root directory called /build. Copy
do-configure-pyrol.sh to /build and modify it to point to your
Trilinos installation. You can also specify the Python interpreter,
library, and include path.

PyROL Installation
------------------

The PyROL module will install by default in a subdirectory of the root
directory called /bin. You can set an alternate install directory by 
setting the flag PYROL\_INSTALL in do-configure-pyrol.sh. Alternatively,
you can let distutils build the module directly in the build directory
by setting the flag PYROL\_BUILD\_INPLACE:BOOL=ON 




