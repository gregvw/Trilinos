import os, sys, subprocess
from setuptools import setup, Extension
import numpy

if "TRILINOS_DIR" not in os.environ:
    print("Please set the TRILINOS_DIR environment variable to point to the Trilinos")
    print("installation directory containing ROL and Teuchos\n")
    sys.exit(1)

trilinos_dir = os.environ["TRILINOS_DIR"]

if (sys.platform == 'darwin'):
    os.environ.setdefault('LDFLAGS','')
    os.environ['LDFLAGS'] += os.path.expandvars(' -Wl,-rpath,$TRILINOS_DIR/lib')

# Make sure we are using c++11
ex_flags = ["-std=c++11"]

if not os.path.exists(trilinos_dir+"/include/ROL_config.h"):
    print("Cannot find Trilinos include directory with ROL. Make sure TRILINOS_DIR is set correctly.\n")
    sys.exit(1)

ex_libs = ["teuchoscore", "teuchosnumerics", "teuchosparameterlist", "teuchoscomm"]

setup(name="PyROL",
      version="0.1.1",
      author="Greg von Winckel, Chris Richardson and Florian Wechsung",
      author_email="pyrol-dev@googlemail.com",
      license="LGPLv3",
#      packages=['pyrol'],
      ext_modules=[
          Extension("pyrol",
                    ["src/PyROL.cpp", "src/PyROL_PythonVector.cpp", "src/PyROL_Objective.cpp",
                     "src/PyROL_EqualityConstraint.cpp", "src/PyROL_NumPyVector.cpp", "src/PyROL_Test.cpp"],
                    include_dirs = [numpy.get_include()] + ["include"] + [trilinos_dir+"/include"],
                    library_dirs = [trilinos_dir+"/lib"],
                    runtime_library_dirs = [trilinos_dir+"/lib"],
                    libraries = ex_libs,
                    extra_compile_args = ex_flags,
                 )],
      )
