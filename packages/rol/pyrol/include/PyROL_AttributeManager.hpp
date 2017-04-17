#ifndef PYROL_ATTRIBUTEMANAGER_HPP
#define PYROL_ATTRIBUTEMANAGER_HPP

#include "PyROL.hpp"

namespace PyROL {

/** \class PyROL::AttributeManager
    \brief Generic class for keeping track of which methods are implemented
 */

class AttributeManager {

protected:

  using Required      = bool;
  using Implemented   = bool;
  using Name          = std::string;
 
  struct Attribute {
    Name        name;
    Required    req;
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

    for( auto a : attrList ) {
      PyObject*   name = PyString_FromString(C_TEXT(a.name));
      Required    req  = a.req;
      Implemented impl = PyObject_HasAttr( pyObj, name ); 

      TEUCHOS_TEST_FOR_EXCEPTION( req & !impl, std::logic_error,
        "Error: The class " << className << " must implement the method " << name ); 

      method_[a.name].name = name;
      method_[a.name].req  = req;
      method_[a.name].impl = impl;
    }
  }
  
  virtual ~AttributeManager() {
    for( auto &m : method_ ) {
      Py_DECREF( m.second.name );
    }
  }

}; // class AttributeManager

} // namespace PyROL 

#endif // PYROL_ATTRIBUTEMANAGER_HPP
