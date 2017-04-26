#ifndef PYROL_ATTRIBUTEMANAGER_HPP
#define PYROL_ATTRIBUTEMANAGER_HPP

#include "PyROL.hpp"

namespace PyROL {

/** \class PyROL::AttributeManager
    \brief Generic class for keeping track of which methods are implemented
 */

class AttributeManager {

protected:

  using Required      = int;
  using Implemented   = int;
  using Name          = const char*;

  struct Attribute {
    Name        name;
    Required    req;

    Attribute( Name n, Required r ) : name(n), req(r) {}
  };

  struct Method {
    PyObject*   name;
    Required    req;
    Implemented impl;
  };

  using AttributeList = std::vector<Attribute>;

  mutable std::map<Name,Method> method_;

public: 

  AttributeManager( PyObject* pyObj, const AttributeList &attrList, 
    const Name &className ) {

    TEUCHOS_TEST_FOR_EXCEPTION( pyObj == NULL, std::logic_error, 
      "Error: attempted to create " << className << " object "
      "from a NULL PyObject* " );

   for( auto a : attrList ) {
      Implemented impl = 0;
      try{
        impl = PyObject_HasAttrString( pyObj, a.name ); 
      }
      catch( std::logic_error err) {
        std::cout << err.what() << std::endl;
      }
      TEUCHOS_TEST_FOR_EXCEPTION( a.req & !impl, std::logic_error,
        "Error: " << className << " must implement the method " << a.name ); 

      method_[a.name].name = PyString_FromString(a.name);
      method_[a.name].req  = a.req;
      method_[a.name].impl = impl;
    }
  }
  
  virtual ~AttributeManager() {
    for( auto &m : method_ ) {
      Py_XDECREF( m.second.name );
    }
  }

#ifdef PYROL_DEBUG_MODE 

  void dir(PyObject* pyObj, std::ostream &os) const {
    PyObject* pyDir = PyObject_Dir(pyObj);

    Py_ssize_t size = PyList_Size(pyDir);
  
    for( Py_ssize_t i=0; i<size; ++i ) {
      PyObject* pyItem = PyList_GetItem(pyDir,i);
      std::string item = PyString_AsString(pyItem);
      os << item << ", ";
    }
    os << std::endl;
    Py_DECREF(pyDir);
  }

#endif

}; // class AttributeManager

} // namespace PyROL 

#endif // PYROL_ATTRIBUTEMANAGER_HPP
