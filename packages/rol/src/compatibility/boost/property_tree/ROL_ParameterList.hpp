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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;


/*  Implements a unified ParameterList interface which conforms to that of
    Teuchos::ParameterList while using the Boost::property_tree implementation.

    This interface only implements a small subset of the available methods,
    however, at the time of creation these amount to all of the functionality
    that ROL has needed.
 */

namespace ROL {

namespace details {


using namespace std;


class ParameterList {
private:
  ROL::SharedPointer<pt::ptree> tree_;

public:
  ParameterList() : tree_(ROL::makeShared<pt::ptree>()) {
  }

  ParameterList( ROL::SharedPointer<pt::ptree> t ) : tree_(t) {
  }

  virtual ~ParameterList() {
  }

  template<class T>
  void set( const string& name, const T& value ) {
    tree_->put(name,value);
  }

  template<class T>
  T get( const string& name ) const {
    return tree_->get<T>(name);
  }

  template<class T>
  T get( const string& name, const T& default_value ) const {
    return tree_->get(name,default_value);
  }

  ParameterList& sublist(const string& name) const {
    pt::ptree& child = tree_->get_child(name);
    auto sublist = ROL::makeSharedFromRef<pt::ptree>(child);
    return *ROL::makeShared<ParameterList>(sublist);
  }

  friend void readParametersFromXml( const string&, ParameterList& parlist );


};

} // namespace details

using ParameterList = details::ParameterList;

inline void readParametersFromXml( const std::string& filename,
                                   ParameterList& parlist ) {

  boost::property_tree::read_xml(filename,*(parlist.tree_));
}

} // namespace ROL
