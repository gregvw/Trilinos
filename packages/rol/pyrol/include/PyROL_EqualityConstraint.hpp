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

#ifndef PYROL_EQUALITYCONSTRAINT_HPP
#define PYROL_EQUALITYCONSTRAINT_HPP

#include "PyROL_AttributeManager.hpp"

namespace PyROL {

/** \class PyROL::EqualityConstraint
    \brief Provides a ROL interface for equality constraint classes implemented in Python
*/

class EqualityConstraint : public ROL::EqualityConstraint<double>, public AttributeManager {

  using Vector = ROL::Vector<double>;

public:

  const static AttributeManager::Name className_;

private:

  const static AttributeManager::AttributeList attrList_;

  PyObject* pyEqCon_;


public:

  EqualityConstraint( PyObject* pyEqCon );
  virtual ~EqualityConstraint();
  virtual void value(Vector &c, const Vector &x, double &tol);
  virtual void applyJacobian(Vector &jv, const Vector &v, const Vector &x, double &tol);
  virtual void applyAdjointJacobian(Vector &ajv, const Vector &v, const Vector &x, double &tol);
  virtual void applyAdjointHessian(Vector &ahuv, const Vector &u, const Vector &v, const Vector &x, double &tol);
  virtual std::vector<double> solveAugmentedSystem(Vector &v1,  Vector &v2, const Vector &b1,
                                                   const Vector &b2, const Vector &x, double &tol);
  virtual void applyPreconditioner(Vector &pv, const Vector &v, const Vector &x, const Vector &g, double &tol);
  virtual void update( const Vector &x, bool flag = true, int iter = -1 );
  virtual bool isFeasible( const Vector &v );

}; // class EqualityConstraint


} // namespace PyROL



#endif // PYROL_EQUALITYCONSTRAINT_HPP
