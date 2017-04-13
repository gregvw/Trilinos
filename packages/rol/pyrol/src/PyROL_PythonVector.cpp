#include "PyROL_PythonVector.hpp"

namespace PyROL {

const AttributeManager::AttributeList PythonVector::attrList_ = {
     //               method name, required
     std::make_tuple( "plus",        false    ),
     std::make_tuple( "scale",       false    ), 
     std::make_tuple( "dot",         false    ), 
     std::make_tuple( "norm",        false    ),
     std::make_tuple( "clone",       true     ),
     std::make_tuple( "axpy",        false    ),
     std::make_tuple( "zero",        false    ),
     std::make_tuple( "basis",       false    ),
     std::make_tuple( "dimension",   false    ), 
     std::make_tuple( "set",         false    ),
     std::make_tuple( "dual",        false    ),
     std::make_tuple( "applyUnary",  false    ),
     std::make_tuple( "applyBinary", false    ),
     std::make_tuple( "reduce",      false    ),
     std::make_tuple( "print",       false    ), 
     std::make_tuple( "checkVector", false    ),
     std::make_tuple( "__getitem__", true     ),  // Overload the [] operator for NumPy-like access
     std::make_tuple( "__setitem__", true     ),  // Overload the [] operator for NumPy-like access
}; 

} // namespace PyROL
