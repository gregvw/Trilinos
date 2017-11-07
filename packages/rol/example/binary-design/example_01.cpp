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

/*! \file  example_01.cpp
    \brief Shows how to minimize a function with binary (0/1) constraints.
*/

#include "ROL_Algorithm.hpp"
#include "ROL_StdVector.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include <iostream>

typedef double RealT;

template<class Real>
class BinaryDesignObjective : public ROL::Objective<Real> {
private:
  const int nvars_;
  const Real alpha_;

public:
  BinaryDesignObjective(const int nvars, const Real alpha)
    : nvars_(nvars), alpha_(alpha) {}

  RealT value(const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();

    Real val(0);
    for (int i=0; i<nvars_; ++i) {
      val += (*ex)[i] + alpha_ * (*ex)[i] * (*ex)[i];
    }
    return val;
  }

  void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    std::shared_ptr<std::vector<Real> > eg
        = dynamic_cast<ROL::StdVector<Real>&>(g).getVector();

    const Real one(1), two(2);
    for (int i=0; i<nvars_; ++i) {
      (*eg)[i] = one + alpha_*two*(*ex)[i];
    }
  }

  void hessVec(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    std::shared_ptr<const std::vector<Real> > ev
        = dynamic_cast<const ROL::StdVector<Real>&>(v).getVector();
    std::shared_ptr<std::vector<Real> > ehv
        = dynamic_cast<ROL::StdVector<Real>&>(hv).getVector();

    const Real two(2);
    for (int i=0; i<nvars_; ++i) {
      (*ehv)[i] = alpha_*two*(*ev)[i];
    }
  }

};

template<class Real>
class BinaryDesignConstraint : public ROL::Constraint<Real> {
private:
  const int nvars_;
  const Real vol_;

public:
  BinaryDesignConstraint(const int &nvars, const Real &vol)
    : nvars_(nvars), vol_(vol) {}

  void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    std::shared_ptr<std::vector<Real> > ec
        = dynamic_cast<ROL::StdVector<Real>&>(c).getVector();

    const Real one(1);
    for (int i=0; i<nvars_; ++i) {
      (*ec)[i] = (*ex)[i] * ((*ex)[i] - one);
    }
    (*ec)[nvars_] = -vol_;
    for (int i=0; i<nvars_; ++i) {
      (*ec)[nvars_] += (*ex)[i];
    }
  }

  void applyJacobian(ROL::Vector<Real> &jv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    std::shared_ptr<const std::vector<Real> > ev
        = dynamic_cast<const ROL::StdVector<Real>&>(v).getVector();
    std::shared_ptr<std::vector<Real> > ejv
        = dynamic_cast<ROL::StdVector<Real>&>(jv).getVector();

    const Real zero(0), one(1), two(2);
    for (int i=0; i<nvars_; ++i) {
      (*ejv)[i] = (two*(*ex)[i]-one) * (*ev)[i];
    }
    (*ejv)[nvars_] = zero;
    for (int i=0; i<nvars_; ++i) {
      (*ejv)[nvars_] += (*ev)[i];
    }
  }

  void applyAdjointJacobian(ROL::Vector<Real> &ajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    std::shared_ptr<const std::vector<Real> > ev
        = dynamic_cast<const ROL::StdVector<Real>&>(v).getVector();
    std::shared_ptr<std::vector<Real> > eajv
        = dynamic_cast<ROL::StdVector<Real>&>(ajv).getVector();

    const Real one(1), two(2);
    for (int i=0; i<nvars_; ++i) {
      (*eajv)[i] = (two*(*ex)[i]-one) * (*ev)[i] + (*ev)[nvars_];
    }
  }

  void applyAdjointHessian(ROL::Vector<Real> &ahuv, const ROL::Vector<Real> &u, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol) {
    std::shared_ptr<const std::vector<Real> > ex
        = dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
    std::shared_ptr<const std::vector<Real> > eu
        = dynamic_cast<const ROL::StdVector<Real>&>(u).getVector();
    std::shared_ptr<const std::vector<Real> > ev
        = dynamic_cast<const ROL::StdVector<Real>&>(v).getVector();
    std::shared_ptr<std::vector<Real> > eahuv
        = dynamic_cast<ROL::StdVector<Real>&>(ahuv).getVector();

    const Real two(2);
    for (int i=0; i<nvars_; ++i) {
      (*eahuv)[i] = two * (*eu)[i] * (*ev)[i];
    }
  }

};

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  int errorFlag  = 0;

  // *** Example body.
 
  try {

    // Set up problem data
    int   dim   = 10; // Set problem dimension. 
    RealT vol   = 2;  // Set desired volume. 
    RealT alpha = 1;  // Set quadratic penalty. 
    std::shared_ptr<std::vector<RealT> > x_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    std::shared_ptr<std::vector<RealT> > g_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    std::shared_ptr<std::vector<RealT> > d_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    std::shared_ptr<std::vector<RealT> > v_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    std::shared_ptr<std::vector<RealT> > jv_rcp = std::make_shared<std::vector<RealT>>(dim+1, 0.0);
    std::shared_ptr<std::vector<RealT> > ajv_rcp = std::make_shared<std::vector<RealT>>(dim, 0.0);
    std::shared_ptr<std::vector<RealT> > c_rcp = std::make_shared<std::vector<RealT>>(dim+1, 0.0);
    for (int i=0; i<dim; i++) {
      (*x_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
      (*g_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
      (*d_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
      (*v_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
      (*ajv_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
    }
    for (int i=0; i<dim+1; i++) {
      (*jv_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
      (*c_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
    }
    std::shared_ptr<ROL::Vector<RealT> > x = std::make_shared<ROL::StdVector<RealT>>(x_rcp);
    std::shared_ptr<ROL::Vector<RealT> > g = std::make_shared<ROL::StdVector<RealT>>(g_rcp);
    std::shared_ptr<ROL::Vector<RealT> > d = std::make_shared<ROL::StdVector<RealT>>(d_rcp);
    std::shared_ptr<ROL::Vector<RealT> > v = std::make_shared<ROL::StdVector<RealT>>(v_rcp);
    std::shared_ptr<ROL::Vector<RealT> > jv = std::make_shared<ROL::StdVector<RealT>>(jv_rcp);
    std::shared_ptr<ROL::Vector<RealT> > ajv = std::make_shared<ROL::StdVector<RealT>>(ajv_rcp);
    std::shared_ptr<ROL::Vector<RealT> > c = std::make_shared<ROL::StdVector<RealT>>(c_rcp);
    std::shared_ptr<ROL::Objective<RealT> > obj = std::make_shared<BinaryDesignObjective<RealT>>(dim, alpha);
    std::shared_ptr<ROL::Constraint<RealT> > con = std::make_shared<BinaryDesignConstraint<RealT>>(dim, vol);

   // Define algorithm
    std::shared_ptr<Teuchos::ParameterList> parlist
      = std::make_shared<Teuchos::ParameterList>();
    std::string paramfile = "input.xml";
    Teuchos::updateParametersFromXmlFile(paramfile,parlist.ptr());
    ROL::Algorithm<RealT> algo("Composite Step",*parlist);

    // Test objective
    obj->checkGradient(*x, *d, true, *outStream);
    *outStream << "\n"; 
    obj->checkHessVec(*x, *v, true, *outStream);
    *outStream << "\n";
    obj->checkHessSym(*x, *d, *v, true, *outStream);
    *outStream << "\n";
    // Test constraint.
    con->checkApplyJacobian(*x, *v, *jv, true, *outStream);
    con->checkAdjointConsistencyJacobian(*jv, *v, *x, true, *outStream);
    con->checkApplyAdjointHessian(*x, *jv, *v, *v, true, *outStream);

    // Run algorithm
    for (int i=0; i<dim; ++i) {
      (*x_rcp)[i] = 1.234*(i<2);
    }
    for (int i=0; i<dim+1; ++i) {
      (*c_rcp)[i] = 0.0;
    }
    algo.run(*x, *c, *obj, *con, true, *outStream);
    *outStream << "x = [";
    for (int i=0; i<dim; ++i) {
      *outStream << (*x_rcp)[i] << "  ";
    }
    *outStream << "]\n";

  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}


