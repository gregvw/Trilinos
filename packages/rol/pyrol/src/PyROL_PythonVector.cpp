// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#include "PyROL_PythonVector.hpp"


namespace PyROL {

const AttributeManager::Name PythonVector::className_ = "PythonVector";

const AttributeManager::AttributeList PythonVector::attrList_ = {
     //               method name, required
     AttributeManager::Attribute( "plus",        0 ),
     AttributeManager::Attribute( "scale",       0 ), 
     AttributeManager::Attribute( "dot",         0 ), 
     AttributeManager::Attribute( "norm",        0 ),
     AttributeManager::Attribute( "clone",       1  ),
     AttributeManager::Attribute( "axpy",        0 ),
     AttributeManager::Attribute( "zero",        0 ),
     AttributeManager::Attribute( "basis",       0 ),
     AttributeManager::Attribute( "dimension",   0 ), 
     AttributeManager::Attribute( "set",         0 ),
     AttributeManager::Attribute( "dual",        0 ),
     AttributeManager::Attribute( "applyUnary",  0 ),
     AttributeManager::Attribute( "applyBinary", 0 ),
     AttributeManager::Attribute( "reduce",      0 ),
     AttributeManager::Attribute( "print",       0 ), 
     AttributeManager::Attribute( "__getitem__", 1 ),  // Overload the [] operator for NumPy-like access
     AttributeManager::Attribute( "__setitem__", 1 ),  // Overload the [] operator for NumPy-like access
}; 

} // namespace PyROL


