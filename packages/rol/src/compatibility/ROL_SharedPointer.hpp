
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

#pragma once

#include <memory>
#include <type_traits>

/* \file ROL_SharedPointer.hpp
 * \brief Provides unified interface to ROL::SharedPointer and 
 *        Teuchos::RCP for legacy support.
 */

#include <cstddef>
#include <utility>


#ifndef ROL_SHARED_POINTER    // Legacy support

#include "Teuchos_RCP.hpp"

namespace ROL {

template<class T> using SharedPointer = Teuchos::RCP<T>;

Teuchos::ENull nullPointer = Teuchos::null;

}


namespace std {

template<class T>
struct is_pointer<ROL::SharedPointer<T>> : public std::true_type { };

}

namespace ROL {

template<class T, class... Args>
inline 
SharedPointer<T> makeShared( Args&&... args ) {
  return Teuchos::rcp( new T(std::forward<Args>(args)...) );
}

template<class T>
inline
SharedPointer<T> makeSharedFromRef( T& obj ) {
  return Teuchos::rcpFromRef(obj);
}

/*
template<class T>
bool operator == ( const SharedPointer<T>& a, const Teuchos::ENull &b ) {
  return a.is_null();
}

template<class T>
bool operator != ( const ROL::SharedPointer<T>& a, const Teuchos::ENull &b ) {
  return a.nonnull();
}
*/
template< class T, class U > 
inline
SharedPointer<T> staticPointerCast( const SharedPointer<U>& r ) noexcept {
  return Teuchos::rcp_static_cast<T>(r);
}

template< class T, class U > 
inline
SharedPointer<T> constPointerCast( const SharedPointer<U>& r ) noexcept {
  return Teuchos::rcp_const_cast<T>(r);
}

template< class T, class U > 
inline
SharedPointer<T> dynamicPointerCast( const SharedPointer<U>& r ) noexcept {
  return Teuchos::rcp_dynamic_cast<T>(r);
}

} // namespace ROL

/*-------------------------------------------------------------------------------*/
#else // Use C++11 std functions

namespace ROL {

template<class T> using SharedPointer = std::shared_ptr<T>;

using nullPointer = ROL::nullPointer;

template<class T, class... Args>
inline
SharedPointer<T> makeShared( Args&&... args ) {
  return std::make_shared<T>(args...);
}

template<class T>
inline
SharedPointer<T> makeSharedFromRef( T& obj ) {
  return std::shared_ptr<T>(obj,[](void*){});
}

template< class T, class U > 
inline
SharedPointer<T> staticPointerCast( const SharedPointer<U>& r ) noexcept {
  return static_pointer_cast<T>(r);
}

template< class T, class U > 
inline
SharedPointer<T> constPointerCast( const SharedPointer<U>& r ) noexcept {
  return const_pointer_cast<T>(r);
}

template< class T, class U > 
inline
SharedPointer<T> dynamicPointerCast( const SharedPointer<U>& r ) noexcept {
  return dynamic_pointer_cast<T>(r);
}

} // namespace ROL

#endif // ROL_SHARED_POINTER

