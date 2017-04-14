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

- In the PYROL\_HOME directory:

mkdir build
cmake ..
make -jN
make install

If you want to specify a particular Python installation to use, 
make the above cmake call while specifying PYTHON\_LIBRARY and PYTHON\_INCLUDE\_DIR.
You can also specify where PyROL should be installed by setting a path PYROL\_INSTALL.
The default location is a subdirectory \bin from the main directory.





