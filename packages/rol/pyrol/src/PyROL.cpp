#include <Python.h>

#include <iostream>

// Basic placeholder code to verify that CMake and Python are 
// playing nice 


static PyObject * 
display( PyObject *self, PyObject *args ) {
  char *myString;
   if( !PyArg_ParseTuple(args,"s",&myString) )
     return NULL;
   std::cout << myString << std::endl;
   Py_INCREF(Py_None);
   return Py_None;
}

static char display_doc[] = 
  "display( ): Output supplied string to console.\n";

static PyMethodDef pyrol_methods[] = {
  {"display", (PyCFunction)display,METH_VARARGS,display_doc},
  {NULL, NULL, 0, NULL}
};

static char pyrol_doc[] = 
  "PyROL: the Python interface to the Rapid Optimization Library";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pyrol_module = {
  PyModuleDef_HEAD_INIT,
  "pyrol",
  pyrol_doc,
  -1,
  pyrol_methods
};
#endif 


#if PY_MAJOR_VERSION < 3
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
#endif 

PyMODINIT_FUNC 
#if PY_MAJOR_VERSION >= 3
PyInit_pyrol(void) {
  return PyModule_Create(&pyrol_module);
}
#else
initpyrol(void) {
  Py_InitModule3("pyrol",pyrol_methods,pyrol_doc);
}
#endif



