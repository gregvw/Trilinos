#ifndef PYROL_ATTRIBUTEMANAGER_HPP
#define PYROL_ATTRIBUTEMANAGER_HPP

#include<string>
#include<tuple>
#include<vector>

namespace PyROL {

/** \class PyROL::AttributeManager
    \brief Generic class for keeping track of which methods are implemented
 */

class AttributeManager {

protected:

  using Required      = bool;
  using Implemented   = bool;
  using Name          = std::string;
  using Attribute     = std::tuple<Name,Required>;
  using Method        = std::tuple<PyObject*,Required,Implemented>;
  using AttributeList = std::vector<Attribute>;

  static const int NAME           = 0;
  static const int IS_REQUIRED    = 1;
  static const int IS_IMPLEMENTED = 2;

  std::map<Name,Method> method_;

  PyObject* pyMethod( const Method &m ) const {
    return std::get<NAME>(m);
  } 

  bool is_implemented( const Method &m ) const {
    return std::get<IS_IMPLEMENTED>(m);
  }

  Method getMethod( const Name &name ) const {
    return method_[name];
  }

public: 

  AttributeManager( PyObject* pyObj, const AttributeList &attrList, 
    const Name &className ) {

    for( auto a : attrList ) {
      Name        name = std::get<NAME>(a);
      Required    req  = std::get<IS_REQUIRED>(a);
      Implemented impl = PyObject_HasAttr( pyObj, name ); 

      TEUCHOS_TEST_FOR_EXCEPTION( req & !impl, std::logic_error,
        "Error: The class " << className << " must implement the method " << name ); 

      method_[name] = std::make_tuple( PyString_FromSring(name), req, impl );
    }
  }
  
  virtual ~AttributeManager() {
    for( auto &m : method_ ) {
      Py_DECREF( pyMethod(m) );
    }
  }

}; // class AttributeManager

} // namespace PyROL 

#endif // PYROL_ATTRIBUTEMANAGER_HPP
