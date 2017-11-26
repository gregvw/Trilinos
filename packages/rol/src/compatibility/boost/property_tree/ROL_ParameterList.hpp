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

#include "ROL_SharedPointer.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;


/*  Implements a unified ParameterList interface which conforms to that of
    ROL::ParameterList while using the Boost::property_tree implementation.

    This interface only implements a small subset of the available methods,
    however, at the time of creation these amount to all of the functionality
    that ROL has needed.
 */

namespace ROL {

namespace details {


using namespace std;

class ParameterList {
private:
  pt::ptree tree_;

  template<class T> static std::string value_type()
  {
    return "";
  }

public:
  std::vector<ParameterList> refs_;

  ParameterList() {
  }

  ParameterList( pt::ptree r ) : tree_(r)
  {
  }

  // FIXME: what to do with "name"
  ParameterList( const string& name)  {
  }

  virtual ~ParameterList() {
  }

  using ConstIterator = pt::ptree::const_iterator;

  ConstIterator begin() const {
    return tree_.begin();
  }

  ConstIterator end() const {
    return tree_.end();
  }

  std::string name(ConstIterator& it ) const {
    // FIXME
    return "name";
  }

  template<class T>
  bool isType( const string& name ) const {
    // FIXME
    return true;
  }

  template<class T>
  void set( const string& name, const T& value ) {
    // Look for existing parameter
    for (auto q : tree_)
    {
      if (q.first == "Parameter" and
          q.second.get<string>("<xmlattr>.name") == name)
      {
        q.second.put("<xmlattr>.value", value);
        std::cout << "Set " << name << " = " << value << "\n";
        return;
      }
    }

    // Make new parameter
    tree_.put("Parameter.<xmlattr>.name", name);
    //    tree_.put("Parameter.<xmlattr>.type", value_type<T>());
    tree_.put("Parameter.<xmlattr>.value", value);
    std::cout << "Set NEW " << name << " = " << value << "\n";

  }

  template<class T>
  T get( const string& name ) const {
    for (auto q : tree_)
    {
      pt::ptree& sub = q.second;
      for (auto r : sub)
      {
        if (r.first == "Parameter")
        {
          const std::string xml_name = r.second.get_child("<xmlattr>").get<std::string>("name");
          if (xml_name == name)
          {
            std::cout << "Found Parameter" << name << "\n";
            T value = r.second.get_child("<xmlattr>").get<T>("value");
            return value;
          }
        }
      }
    }
    return T();
  }

  std::string get( const string& name, const string& default_value) const {
    return get<string>(name);
  }

  template<class T>
  T get( const string& name, const T& default_value ) const {
    return get<T>(name);
  }

  void print(pt::ptree& r, std::string indent="")
  {
    for (auto q : r)
    {
      pt::ptree& sub = q.second;
      if (sub.size() == 0)
      {
        std::cout << indent << "[" << q.first << "] = \"";
        std::cout << r.get<std::string>(q.first) << "\"\n";
      }
      else
      {
        std::cout << indent << "[" << q.first << "]\n";
        print(sub, indent + "  ");
      }
    }
  }

  ParameterList& sublist(const string& name) {
    //    std::cout << "======= " << name << "\n";

    //    print(tree_);
    for (auto r : tree_)
    {
      if (r.first == "ParameterList")
      {
        const std::string xml_name = r.second.get_child("<xmlattr>").get<std::string>("name");
        if (xml_name == name)
        {
          std::cout << "Found " << name << "\n";
          refs_.push_back(ParameterList(r.second));
          return refs_.back();
        }
      }
    }
    std::cout << "Failed to find " << name << "\n";

    return *this;
  }

  bool isSublist(const string& name) const
  {
    auto it = tree_.find(name);
    if (it == tree_.not_found())
      return false;
    // FIXME: check it is a ptree and not a regular parameter
    return true;
  }

  bool isParameter(const string& name) const
  {
    auto it = tree_.find(name);
    if (it == tree_.not_found())
      return false;
    // FIXME: check it is a regular parameter and not a ptree
    return true;
  }

  pt::ptree& tree()
  { return tree_; }

  //  friend void readParametersFromXml( const string&, ParameterList& parlist );


};

} // namespace details

  using ParameterList = details::ParameterList;

  template <class T>
  inline std::vector<T> getArrayFromStringParameter(const ParameterList& parlist,
                                          const std::string& name)
  {
    std::string p = parlist.get<std::string>(name);
    std::vector<std::string> p_split;
    boost::split(p_split, p, boost::is_any_of(","));

    std::vector<T> result;
    for (auto &q : p)
      result.push_back(boost::lexical_cast<T>(q));

    return result;
  }

  inline ROL::SharedPointer<ParameterList> getParametersFromXmlFile( const std::string& filename )
  {
    pt::ptree tr;
    boost::property_tree::read_xml(filename, tr);
    auto list = ROL::makeShared<ParameterList>(tr.get_child("ParameterList"));
    return list;
  }

  inline void readParametersFromXml( const std::string& filename,
                                     ParameterList& parlist ) {
    boost::property_tree::read_xml(filename, parlist.tree());
  }

  inline void updateParametersFromXmlFile( const std::string& infile, ParameterList& inlist )
  {
    // FIXME: do something
  }

  inline void writeParameterListToXmlFile( ParameterList& parlist,
                                           const std::string& filename ) {
    boost::property_tree::write_xml(filename, parlist.tree());
  }

} // namespace ROL
